"""Scripted-clock engineering checks; no scientific case or world generation."""
import json

import pytest
import torch

import openjev.research.finite_timed_prefix as timed
from openjev.research.finite_expected_count_bridge import DYNAMICS, fit_prefix
from openjev.research.finite_factorized_dynamics_models import make_model

SEED = 937101


class ScriptedClock:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        assert self.calls < len(self.values), 'unexpected additional clock read/update'
        value = self.values[self.calls]
        self.calls += 1
        return value


def public():
    prefix = torch.zeros((2, 9, 31), dtype=torch.float32)
    lengths = torch.tensor([9, 3], dtype=torch.int64)
    for row, length in enumerate(lengths.tolist()):
        prefix[row, 0, 4 + row] = prefix[row, 0, 9] = 1
        for step in range(1, length):
            prefix[row, step, (step + row) % 4] = 1
            prefix[row, step, 8 if row == 1 and step == 2 else 4 + step % 4] = 1
    return prefix, lengths


def state(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def equal_parameters(first, second):
    assert set(first) == set(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_exact_boundary_accepts_once_and_charges_postcommit_time():
    model = make_model('factorized', SEED)
    head = model.cost_logits.detach().clone()
    clock = ScriptedClock([100., 101., 110., 110.5, 111.])
    result = timed.timed_fit(model, *public(), 'gradient', 10., clock=clock)
    assert clock.calls == 5 and result['status'] == 'PASS'
    assert result['accepted_updates'] == result['attempted_updates'] == 1
    assert result['termination'] == 'budget_reached'
    assert result['trace'][0]['completed_elapsed'] == 10
    assert result['trace'][0]['accepted'] is True and result['trace'][0]['rolled_back'] is False
    assert result['timed_seconds'] == 10.5 and result['overrun_seconds'] == .5
    assert result['final_summary_seconds'] == .5
    assert torch.equal(head, model.cost_logits)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('method', ['em', 'gradient'])
def test_late_first_update_restores_every_parameter_and_charges_attempt(method):
    model = make_model('factorized', SEED)
    original = state(model)
    result = timed.timed_fit(model, *public(), method, 10.,
                             clock=ScriptedClock([0., 1., 10.1, 10.2, 10.4]))
    equal_parameters(state(model), original)
    assert result['status'] == 'PASS' and result['termination'] == 'late_update_rolled_back'
    assert result['accepted_updates'] == 0 and result['attempted_updates'] == 1
    row = result['trace'][0]
    assert not row['accepted'] and row['rolled_back']
    assert row['dynamics_before_sha256'] == row['dynamics_retained_sha256'] == result['final_dynamics_sha256']
    assert result['initial_dynamics_sha256'] == result['final_dynamics_sha256']
    assert result['final_penalized_log_likelihood'] == result['initial_penalized_log_likelihood']
    assert result['work']['parameter_snapshot_tensors'] == result['work']['parameter_restore_tensors'] == 3
    assert result['work']['parameter_snapshot_bytes'] == result['work']['parameter_restore_bytes'] == 2560
    assert result['timed_seconds'] == 10.2 and result['overrun_seconds'] == pytest.approx(.2)
    if method == 'em':
        assert result['work']['numpy_expectation_passes'] == 2
        assert result['work']['map_updates'] == result['work']['probability_imports'] == 1
        assert result['work']['expectation_event_exposures'] == 2 * 12
    else:
        assert result['work']['full_gradient_passes'] == result['work']['adam_updates'] == 1
        assert result['late_optimizer_state_discarded'] is True


def test_two_accepted_gradient_steps_match_fixed_bridge_persistent_adam():
    model, reference = make_model('factorized', SEED), make_model('factorized', SEED)
    fit_prefix(reference, *public(), 'gradient', 2)
    result = timed.timed_fit(model, *public(), 'gradient', 10.,
                             clock=ScriptedClock([0., 1., 2., 3., 4., 10., 10.2, 10.5]))
    equal_parameters(state(model), state(reference))
    assert result['accepted_updates'] == result['attempted_updates'] == 2
    assert result['optimizer_state_reused_across_steps'] is True
    assert result['work']['adam_updates'] == 2 and result['work']['torch_diagnostic_passes'] == 3
    assert result['trace'][0]['dynamics_retained_sha256'] == result['trace'][1]['dynamics_before_sha256']
    assert result['work']['parameter_restore_tensors'] == 0


def test_late_second_update_restores_first_accepted_not_initial_state():
    model, reference = make_model('factorized', SEED), make_model('factorized', SEED)
    fit_prefix(reference, *public(), 'gradient', 1)
    result = timed.timed_fit(model, *public(), 'gradient', 10.,
                             clock=ScriptedClock([0., 1., 2., 3., 10.1, 10.2, 10.5]))
    equal_parameters(state(model), state(reference))
    assert result['accepted_updates'] == 1 and result['attempted_updates'] == 2
    assert result['trace'][0]['accepted'] and not result['trace'][1]['accepted']
    assert result['trace'][0]['dynamics_retained_sha256'] == result['final_dynamics_sha256']
    assert result['initial_dynamics_sha256'] != result['final_dynamics_sha256']


def test_setup_exhausting_budget_never_starts_an_update():
    model = make_model('factorized', SEED)
    original = state(model)
    result = timed.timed_fit(model, *public(), 'gradient', 10.,
                             clock=ScriptedClock([0., 10.2, 10.3, 10.5]))
    equal_parameters(state(model), original)
    assert result['trace'] == [] and result['accepted_updates'] == result['attempted_updates'] == 0
    assert result['work']['adam_updates'] == result['work']['parameter_snapshot_tensors'] == 0
    assert result['work']['torch_diagnostic_passes'] == 1


def test_secondary_cap_before_deadline_is_preserved_failure(monkeypatch):
    monkeypatch.setattr(timed, 'MAX_UPDATES', 1)
    model = make_model('factorized', SEED)
    result = timed.timed_fit(model, *public(), 'gradient', 10.,
                             clock=ScriptedClock([0., 1., 2., 2.1, 2.3]))
    assert result['status'] == 'FAILED_UPDATE_CAP' and result['termination'] == 'update_cap_before_deadline'
    assert result['accepted_updates'] == result['attempted_updates'] == 1
    assert result['trace'][0]['accepted'] and result['overrun_seconds'] == 0
    assert result['final_dynamics_sha256'] == result['trace'][0]['dynamics_attempted_sha256']


def test_nonmonotone_completion_clock_rejects_and_rolls_back():
    model = make_model('factorized', SEED)
    original = state(model)
    with pytest.raises(ValueError, match='nondecreasing clock'):
        timed.timed_fit(model, *public(), 'gradient', 10., clock=ScriptedClock([0., 2., 1.]))
    equal_parameters(state(model), original)


def test_public_arrays_and_head_are_unmodified_in_em():
    model = make_model('factorized', SEED)
    prefix, lengths = public()
    before_prefix, before_lengths, head = prefix.clone(), lengths.clone(), model.cost_logits.detach().clone()
    result = timed.timed_fit(model, prefix, lengths, 'em', 10., clock=ScriptedClock([0., 1., 10., 10.1, 10.2]))
    assert torch.equal(prefix, before_prefix) and torch.equal(lengths, before_lengths)
    assert torch.equal(head, model.cost_logits)
    assert result['trace'][0]['penalized_log_likelihood_after'] >= result['trace'][0]['penalized_log_likelihood_before'] - 1e-9
    assert result['numpy_work']['sequences'] == 4
    assert result['numpy_work']['found_events'] == 2
    assert set(DYNAMICS).issubset(dict(model.named_parameters()))
