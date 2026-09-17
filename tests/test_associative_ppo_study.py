import copy
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('minigrid')
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('associative_ppo_test', ROOT/'scripts/associative_ppo_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture
def tiny_protocol(monkeypatch):
    p = copy.deepcopy(study.PROTOCOL)
    p.update(evaluation_episodes=4, environments=2, max_steps=2)
    monkeypatch.setattr(study, 'PROTOCOL', p)
    torch.set_num_threads(1)
    return p


@pytest.mark.parametrize('arm,mode', [('gru', 'reset_all'), ('fast_global', 'reset_store'),
                                     ('fast_selective', 'intact'), ('fast_global', 'cue_swapped')])
def test_evaluation_resets_counts_and_telemetry_are_episode_local(tiny_protocol, arm, mode):
    model = study.AssociativePolicy(arm)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.bias[6] = 10  # Actual native no-op, guaranteeing the declared timeout.
    result = study.evaluate(model, arm, 11, mode)
    study.validate_episode_result(result)
    assert result['timeout'] == 1 and result['mean_length'] == 2
    assert all(r['action_counts'] == [0, 0, 0, 0, 0, 0, 2] for r in result['episodes'])
    if model.uses_store:
        visible, hidden = result['write_audit']['cue_visible'], result['write_audit']['cue_hidden']
        assert visible['steps'] == 8 and hidden['steps'] == 0
        assert visible['beta_sum']/visible['steps'] == pytest.approx(.1)
        assert visible['update_norm_sum'] > 0
    else:
        assert result['write_audit'] is None


def valid_result(p):
    start = p['evaluation_seed_start']+110000
    rows = [{'seed': start+i, 'success': False, 'wrong_goal': False, 'timeout': True,
             'length': 2, 'return': 0., 'action_counts': [0, 0, 0, 0, 0, 0, 2]} for i in range(4)]
    return {'size': 11, 'episodes': rows, 'success': 0., 'wrong_goal': 0., 'timeout': 1.,
            'mean_length': 2., 'mean_return': 0.}


@pytest.mark.parametrize('change', ['duplicate_seed', 'nan_return', 'wrong_count', 'wrong_aggregate'])
def test_score_validation_rejects_corrupted_episode_evidence(tiny_protocol, change):
    result = valid_result(tiny_protocol)
    study.validate_episode_result(result)
    if change == 'duplicate_seed':
        result['episodes'][0]['seed'] = result['episodes'][1]['seed']
    elif change == 'nan_return':
        result['episodes'][0]['return'] = float('nan')
    elif change == 'wrong_count':
        result['episodes'][0]['action_counts'][-1] = 1
    else:
        result['success'] = .5
    with pytest.raises(ValueError):
        study.validate_episode_result(result)


def test_continuation_requires_advantage_over_plain_gru_and_store_use():
    averages = {arm: {'intact': {str(size): {'success': .5} for size in [11, 17, 23]}}
                for arm in study.ARMS}
    averages['fast_selective'] = {'intact': {str(size): {'success': .9} for size in [11, 17, 23]},
                                    'reset_store': {'11': {'success': .5}}}
    fits = {'fast_selective': {'intact': {'11': [{'success': .9}] * 3}}}
    assert all(study.gates(averages, fits).values())
    averages['gru']['intact']['17']['success'] = .9
    assert not study.gates(averages, fits)['gain_vs_gru_size_17']
    averages['fast_selective']['reset_store']['11']['success'] = .9
    assert not study.gates(averages, fits)['store_reset_effect']


def test_all_declared_evaluation_conditions_are_counted():
    assert sum(len(study.modes(arm))*3 for arm in study.ARMS)+1 == 55
    assert 'reset_store' not in study.modes('gru')
    assert 'reset_store' in study.modes('fast_selective')
