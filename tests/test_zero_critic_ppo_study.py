import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('minigrid')
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('zero_critic_test', ROOT/'scripts/zero_critic_ppo_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('arm', study.PROTOCOL['arms'])
def test_zero_factory_preserves_initialization_rng_and_trainable_head(arm):
    torch.manual_seed(7)
    original = study.MODEL(arm)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    zero, receipt = study.zero_model(arm)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert zero.value.weight.requires_grad and zero.value.bias.requires_grad
    assert not torch.count_nonzero(zero.value.weight) and not torch.count_nonzero(zero.value.bias)
    assert torch.count_nonzero(original.value.weight)
    for name, value in original.state_dict().items():
        if not name.startswith('value.'):
            assert torch.equal(value, zero.state_dict()[name])
    assert receipt['initial_original_parameters_sha256'] == study.parameter_digest(original.state_dict())
    assert receipt['initial_zero_parameters_sha256'] == study.parameter_digest(zero.state_dict())
    assert receipt['policy_logits_and_states_identical'] and receipt['zero_value_head_trainable']
    assert all(p.grad is None for p in zero.parameters())


def test_scope_restores_original_functions_and_never_mutates_reference():
    names = ['PROTOCOL', 'AssociativePolicy', 'fit', 'signature']
    before = {name: getattr(study.base, name) for name in names}
    reference = {name: getattr(study.reference, name) for name in names}
    with pytest.raises(RuntimeError), study.configured({'sentinel': 'not_accessed'}):
        assert study.base.AssociativePolicy is study.factory
        assert study.base.fit is study.checked_fit
        assert study.base.PROTOCOL is study.PROTOCOL
        assert study.base.signature is not before['signature']
        assert study.reference.PROTOCOL['version'] == 'associative-ppo-v1'
        raise RuntimeError('test interruption')
    for name in names:
        assert getattr(study.base, name) is before[name]
        assert getattr(study.reference, name) is reference[name]


def test_checked_fit_uses_original_function_and_writes_preflight(tmp_path, monkeypatch):
    def stub(arm, seed, out, *, updates=None):
        assert seed == 7 and updates == 2
        torch.manual_seed(seed)
        model = study.base.AssociativePolicy(arm)
        assert not torch.count_nonzero(model.value.weight)
        out.mkdir()
        return {'interactions': 2048, 'gradient_steps': 8}

    monkeypatch.setattr(study, 'REFERENCE_FIT', stub)
    with study.configured():
        before = study.base.AssociativePolicy
        result = study.checked_fit('gru', 7, tmp_path/'smoke', updates=2)
        assert study.base.AssociativePolicy is before
    assert result['gradient_steps'] == 8
    receipt = json.loads((tmp_path/'smoke/initialization.json').read_text())
    assert receipt['arm'] == 'gru' and receipt['seed'] == 7 and receipt['smoke'] is True
    assert receipt['rng_unchanged'] and receipt['only_value_parameters_changed']


def test_checked_fit_restores_inner_factory_when_reference_fit_raises(tmp_path, monkeypatch):
    def interrupted(arm, seed, out, *, updates=None):
        torch.manual_seed(seed)
        model = study.base.AssociativePolicy(arm)
        assert not torch.count_nonzero(model.value.weight)
        raise RuntimeError('simulated optimizer failure')

    monkeypatch.setattr(study, 'REFERENCE_FIT', interrupted)
    with study.configured():
        before = study.base.AssociativePolicy
        with pytest.raises(RuntimeError, match='optimizer failure'):
            study.checked_fit('gru', 7, tmp_path/'failed', updates=2)
        assert study.base.AssociativePolicy is before
    assert not (tmp_path/'failed/initialization.json').exists()


@pytest.mark.parametrize('arm,seed,updates', [('gru', 7, None), ('gru', 7, 1), ('gru', 101, 2),
                                            ('gru', 999, None), ('new_algorithm', 7, 2)])
def test_checked_fit_rejects_shortened_panel_or_undeclared_fit(tmp_path, arm, seed, updates):
    with pytest.raises(ValueError):
        study.checked_fit(arm, seed, tmp_path/'unused', updates=updates)
    assert not (tmp_path/'unused').exists()


def test_preparation_requires_completed_control_before_reading_other_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    paths = {'control_plan': 'control-plan.json', 'control_run': 'control-run',
             'control_report': 'control-report', 'probe_plan': 'probe-plan.json', 'probe_run': 'probe-run'}
    with pytest.raises(ValueError, match='must finish first'):
        study.validate_inputs(paths)


def test_other_training_settings_cannot_change(monkeypatch):
    monkeypatch.setitem(study.PROTOCOL, 'learning_rate', .5)
    with pytest.raises(ValueError, match='other than the value-head'):
        study.signature({})


def fake_summary(success_counts):
    p = study.reference.PROTOCOL
    evaluation, averages, per_fit = [], {}, {}
    for arm in study.PROTOCOL['arms']:
        averages[arm], per_fit[arm] = {}, {}
        for mode in study.reference.modes(arm):
            averages[arm][mode], per_fit[arm][mode] = {}, {}
            for seed in study.PROTOCOL['seeds']:
                results = []
                for size in study.PROTOCOL['eval_sizes']:
                    count = success_counts[arm]
                    first = p['evaluation_seed_start']+size*10000
                    episodes = [{'seed': first+i, 'success': i < count, 'wrong_goal': False,
                                 'timeout': i >= count, 'length': 10,
                                 'return': 1-.9*10/p['max_steps'] if i < count else 0.,
                                 'action_counts': [10, 0, 0, 0, 0, 0, 0]} for i in range(4)]
                    result = {'size': size, 'episodes': episodes, 'success': count/4,
                              'wrong_goal': 0., 'timeout': 1-count/4, 'mean_length': 10.,
                              'mean_return': count/4*(1-.9*10/p['max_steps'])}
                    results.append(result)
                    per_fit[arm][mode].setdefault(str(size), []).append({'success': count/4})
                    averages[arm][mode][str(size)] = {'success': count/4}
                evaluation.append({'arm': arm, 'seed': seed, 'mode': mode, 'results': results})
    random_results = copy.deepcopy(evaluation[0]['results'])
    for result in random_results:
        for row in result['episodes']:
            row.update(success=False, wrong_goal=False, timeout=True, **{'return': 0.})
        result.update(success=0., wrong_goal=0., timeout=1., mean_return=0.)
    evaluation.append({'arm': 'random', 'mode': 'intact', 'results': random_results})
    return {'evaluation': evaluation, 'averages': averages, 'per_fit': per_fit}


@pytest.fixture
def paired_summaries(monkeypatch):
    monkeypatch.setitem(study.reference.PROTOCOL, 'evaluation_episodes', 4)
    return (fake_summary({arm: 1 for arm in study.PROTOCOL['arms']}),
            fake_summary({'gru': 2, 'feedforward': 2, 'fast_global': 2, 'fast_selective': 1}))


def test_optimization_gate_pairs_all_seeds_and_keeps_timeouts(paired_summaries):
    original, zero = paired_summaries
    result = study.comparison(original, zero)
    assert result['optimization_continuation_passed']
    assert sum(result['improved_arms'].values()) == 3
    assert len(result['paired_fits']) == 162
    row = next(r for r in result['paired_fits'] if r['arm'] == 'gru' and r['seed'] == 101
               and r['mode'] == 'intact' and r['size'] == 11)
    assert row['assigned_pairs'] == 4 and row['zero_only_success'] == 1
    assert row['original_only_success'] == 0 and row['success_difference'] == .25
    assert not all(result['zero_selective_checks'].values())
    assert result['independent_confirmation'] is False
    assert result['repeated_random_episode_identity_verified'] is True


def test_optimization_gate_rejects_arm_degradation(paired_summaries):
    original, _ = paired_summaries
    zero = fake_summary({'gru': 2, 'feedforward': 2, 'fast_global': 2, 'fast_selective': 0})
    result = study.comparison(original, zero)
    assert result['optimization_checks']['minimum_improved_arms']
    assert not result['optimization_checks']['no_excessive_degradation']
    assert not result['optimization_continuation_passed']


def test_optimization_gate_requires_three_improved_architectures(paired_summaries):
    original, _ = paired_summaries
    zero = fake_summary({'gru': 2, 'feedforward': 2, 'fast_global': 1, 'fast_selective': 1})
    result = study.comparison(original, zero)
    assert not result['optimization_checks']['minimum_improved_arms']


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'episode_seed'])
def test_comparison_rejects_missing_duplicate_or_unpaired_evidence(paired_summaries, corruption):
    original, zero = copy.deepcopy(paired_summaries)
    if corruption == 'duplicate':
        zero['evaluation'].append(zero['evaluation'][0])
    elif corruption == 'missing':
        zero['evaluation'].pop(0)
    else:
        zero['evaluation'][0]['results'][0]['episodes'][0]['seed'] += 1
    with pytest.raises(ValueError):
        study.comparison(original, zero)


def test_repeated_random_reference_allows_different_timing_only(paired_summaries):
    original, zero = copy.deepcopy(paired_summaries)
    for result in zero['evaluation'][-1]['results']:
        result['wall_seconds'] = 123.
        result['model_forward_seconds'] = .5
    assert study.comparison(original, zero)['repeated_random_episode_identity_verified']


@pytest.mark.parametrize('corruption', ['duplicate', 'missing_size', 'mode', 'episode_actions'])
def test_repeated_random_reference_rejects_changed_evidence(paired_summaries, corruption):
    original, zero = copy.deepcopy(paired_summaries)
    record = zero['evaluation'][-1]
    if corruption == 'duplicate':
        zero['evaluation'].append(record)
    elif corruption == 'missing_size':
        record['results'].pop()
    elif corruption == 'mode':
        record['mode'] = 'reset_all'
    else:
        # Counts still sum to the valid episode length, but the repeated policy
        # must have exactly the same action counts as the original reference.
        record['results'][0]['episodes'][0]['action_counts'] = [9, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match='random|Random'):
        study.comparison(original, zero)
