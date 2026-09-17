"""Gymnasium firing-control task with causal history and explicit finite horizon."""
import time
from typing import ClassVar

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from openjev.domain import hard_decision, teacher_action
from openjev.game import Doom
from openjev.research.memory import DecisionHistory


def current_features(obs):
    return np.array([1., float(obs.visible),
                     min(abs(obs.aim_error)/max(.04, .65*obs.half_width), 3.)/3.,
                     obs.half_width, min(obs.distance/1000., 2.),
                     np.clip(obs.health/100., 0., 2.), min(obs.ammo/30., 2.)])


def rl_features(obs, history, kind, steps, horizon):
    current = current_features(obs)
    # Equal input and network dimensions; current-only has no history information.
    temporal = history.features(current)[7:] if kind == 'history' else np.zeros(10)
    return np.r_[current, temporal, (horizon-steps)/horizon].astype(np.float32)


class FiringEnv(gym.Env):
    """RL chooses fire/wait; all arms retain the same rule steering."""
    metadata: ClassVar[dict] = {'render_modes': []}

    def __init__(self, kind='history', scenario='defend_the_center', seed_start=200000,
                 horizon=180, tics=7, trace=None, label=None):
        super().__init__()
        if kind not in ('current','history') or horizon <= 0:
            raise ValueError('Invalid features or horizon')
        self.kind, self.scenario = kind, scenario
        self.seed_start, self.horizon, self.tics = seed_start, horizon, tics
        self.trace, self.label = trace, label or {}
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-10.,10.,shape=(18,),dtype=np.float32)
        self.doom = None
        self.episode_index = 0
        self.training_episodes = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.game_seed = (options or {}).get('game_seed',self.seed_start+self.episode_index)
        self.episode_index += 1
        self.doom = Doom(self.scenario,self.game_seed)
        self.obs = self.doom.observe()
        self.history = DecisionHistory()
        self.steps = self.fired = self.success = 0
        self.kills = 0
        self.return_ = 0.
        self.decision_seconds = []
        self.initial_ammo = self.obs.ammo
        self.done = False
        return rl_features(self.obs,self.history,self.kind,0,self.horizon), {}

    def step(self, action):
        if self.done or self.doom is None:
            raise RuntimeError('Reset before stepping a finished episode')
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError('Expected fire/wait action')
        before_obs = self.obs
        start = time.perf_counter()
        x = current_features(before_obs)
        steer, _ = teacher_action(before_obs)
        decision = hard_decision(steer,bool(action),'rl')
        issued = bool(action and before_obs.ammo > 0 and before_obs.directive != 'pacifist')
        compute = time.perf_counter()-start
        before_hit = self.doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        self.doom.step(decision,before_obs,self.tics)
        hit = max(0.,self.doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)-before_hit)
        stats = self.doom.stats()
        kills = stats['kills']-self.kills
        self.kills = stats['kills']
        self.steps += 1
        self.fired += issued
        self.success += int(issued and hit > 0)
        # A finite-horizon objective: remaining horizon is observed and the cap is
        # a terminal state, not an artificial timeout requiring value bootstrap.
        terminated = stats['finished'] or self.steps >= self.horizon
        reward = kills - float(stats['dead']) - .01*issued
        self.return_ += reward
        self.obs = self.doom.observe()
        start = time.perf_counter()
        self.history.update(x,issued,hit)
        feature = (rl_features(self.obs,self.history,self.kind,self.steps,self.horizon)
                   if self.obs is not None else np.zeros(18,dtype=np.float32))
        compute += time.perf_counter()-start
        self.decision_seconds.append(compute)
        self.done = terminated
        info = {'issued_fire':issued,'hit_count_change':hit,'decision_seconds':compute,
                'game_seed':self.game_seed,'capped':not stats['finished'] and terminated}
        if self.trace:
            import json
            self.trace.write(json.dumps({**self.label,'scenario':self.scenario,'seed':self.game_seed,
                'step':self.steps-1,'observation':before_obs.to_dict(),'issued_fire':issued,
                'hit_count_change':hit,'kill_change':kills,'reward':reward,
                'terminated':terminated,'capped':info['capped']})+'\n')
        if terminated:
            info['result'] = self.result(stats)
            self.training_episodes.append(info['result'])
        return feature,float(reward),terminated,False,info

    def result(self, stats=None):
        stats = stats or self.doom.stats()
        return {'scenario':self.scenario,'seed':self.game_seed,'steps':self.steps,
                'kills':stats['kills'],'game_seconds':stats['game_seconds'],
                'engine_reward':stats['reward'],'training_reward':self.return_,
                'firing_windows':self.fired,'success_windows':self.success,
                'utility':self.success-.25*self.fired,'ammo_used':self.initial_ammo-stats['ammo'],
                'capped':not stats['finished'],'dead':stats['dead']}

    def close(self):
        if self.doom is not None:
            self.doom.close()
            self.doom = None
