import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="Install the rl extra for RL environment tests")

from openjev.domain import Observation
from openjev.research.memory import DecisionHistory
from openjev.research.rl_env import FiringEnv, current_features, rl_features


def test_history_is_causal_and_current_padding_is_empty():
    obs = Observation(visible=True,aim_error=.1,half_width=.1,ammo=20)
    h = DecisionHistory()
    initial = rl_features(obs,h,'history',0,180)
    assert initial[13]==0 and initial[15]==0 and initial[16]==0
    h.update(current_features(obs),True,1)
    changed = rl_features(obs,h,'history',1,180)
    assert changed[13]==1 and changed[15]==1 and changed[16]==1
    np.testing.assert_array_equal(rl_features(obs,h,'current',1,180)[7:17],np.zeros(10))
    assert changed[-1]==pytest.approx(179/180)


def test_engine_cap_terminates_and_reset_clears_history():
    env=FiringEnv(horizon=1,seed_start=151000)
    try:
        state,_=env.reset()
        assert env.observation_space.contains(state)
        state,_,terminated,truncated,info=env.step(1)
        assert terminated and not truncated and info['capped']
        assert info['result']['steps']==1
        with pytest.raises(RuntimeError):
            env.step(0)
        state,_=env.reset()
        assert state[13]==0 and state[15]==0 and state[16]==0 and state[-1]==1
    finally:
        env.close()


def test_reward_tracks_kills_and_death_not_hit_windows():
    env=FiringEnv(horizon=20,seed_start=151010)
    try:
        env.reset()
        total=0
        while True:
            _,r,done,_,info=env.step(1)
            total+=r
            if done:
                break
        result=info['result']
        assert total==pytest.approx(result['kills']-result['dead']-.01*result['firing_windows'])
    finally:
        env.close()
