import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research.policy_variants import actor_loss, advantages, reduce_steps
from openjev.research.video_reward import pixel_embedding, prepare_frame, sample_clip


def test_reward_baselines_and_scaling():
    rewards = np.array([0., 0., 1.])
    np.testing.assert_allclose(advantages(rewards, 'rloo'), [-.5, -.5, 1.])
    np.testing.assert_allclose(advantages(rewards, 'dr_grpo'), rewards-rewards.mean())
    np.testing.assert_allclose(advantages(rewards, 'grpo'), advantages(rewards, 'dapo_clip_loss'))
    np.testing.assert_allclose(advantages(10*rewards, 'dr_grpo'), 10*advantages(rewards, 'dr_grpo'))
    for variant in ['grpo', 'dr_grpo', 'dapo_clip_loss', 'rloo']:
        assert not advantages([1., 1., 1.], variant).any()


def test_variable_length_normalization():
    values = torch.tensor([2., 4., 4., 4.])
    lengths = [1, 3]
    assert reduce_steps(values, lengths, 'grpo', 4) == 3.
    assert reduce_steps(values, lengths, 'dapo_clip_loss', 4) == 3.5
    assert reduce_steps(values, lengths, 'dr_grpo', 4) == 1.75
    assert reduce_steps(values, lengths, 'rloo', 4) == 1.75
    with pytest.raises(ValueError):
        reduce_steps(values, [4, 1], 'grpo', 4)


class Policy:
    def __init__(self):
        self.logits = torch.tensor([[0., 0.]], requires_grad=True)

    def get_distribution(self, states):
        from types import SimpleNamespace
        return SimpleNamespace(distribution=torch.distributions.Categorical(logits=self.logits.expand(len(states), -1)))


def test_asymmetric_clip_and_rloo_gradient():
    policy = Policy()
    x = torch.zeros((2, 1))
    actions = torch.zeros(2, dtype=torch.int64)
    logs = torch.full((2,), np.log(.4), dtype=torch.float32)  # ratio 1.25
    ref = torch.full((2, 2), .5)
    adv = torch.ones(2)
    args = (policy, x, actions, logs, ref, adv, [1, 1])
    grpo, _ = actor_loss(*args, 'grpo', horizon=1, kl_weight=0., entropy_weight=0.)
    dapo, _ = actor_loss(*args, 'dapo_clip_loss', horizon=1, kl_weight=0., entropy_weight=0.)
    assert float(grpo.detach()) == pytest.approx(-1.2)
    assert float(dapo.detach()) == pytest.approx(-1.25)
    grpo.backward()
    assert policy.logits.grad.abs().sum() == 0
    policy.logits.grad = None
    rloo, _ = actor_loss(*args, 'rloo', horizon=1, kl_weight=0., entropy_weight=0.)
    rloo.backward()
    assert policy.logits.grad[0, 0] < 0  # gradient descent increases rewarded action


def test_kl_penalizes_departure():
    policy = Policy()
    x = torch.zeros((2, 1))
    args = (policy, x, torch.zeros(2, dtype=torch.int64), torch.full((2,), np.log(.5)),
            torch.tensor([[.9, .1], [.9, .1]]), torch.zeros(2), [1, 1], 'grpo')
    loss, metrics = actor_loss(*args, horizon=1, kl_weight=1., entropy_weight=0.)
    assert loss > 0 and metrics['kl'] > 0
    loss.backward()
    assert policy.logits.grad[0, 0] < 0


def test_video_hud_removed_and_sampling_reproducible():
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    clean = prepare_frame(frame)
    frame[400:] = 255
    np.testing.assert_array_equal(clean, prepare_frame(frame))
    clip, indices = sample_clip([clean, clean+1, clean+2], 16)
    assert clip.shape == (16, 256, 256, 3)
    assert indices[0] == 0 and indices[-1] == 2
    embedding = pixel_embedding(clip)
    assert embedding.shape == (192,)
    assert np.linalg.norm(embedding) == pytest.approx(1.)


def test_sparse_env_terminal_only_and_actor_transfer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'))
    import json

    from jepa_rl_doom import PROTOCOL, ROOT, SparseGoalEnv, make_actor_critic
    from stable_baselines3 import PPO
    p = json.loads(PROTOCOL.read_text())
    env = SparseGoalEnv(horizon=2, seed_start=199901, success_kills=0)
    try:
        env.reset()
        assert env.step(0)[1] == 0.
        assert env.step(0)[1:3] == (1., True)
        initial = PPO.load(ROOT/p['initial_run']/'ppo_history-0.zip', device='cpu')
        models = [make_actor_critic(a, env, initial, p, 0) for a in ['ppo_sparse', 'a2c_sparse']]
        for key, value in initial.policy.state_dict().items():
            if key.startswith(('mlp_extractor.policy_net.', 'action_net.')):
                assert all(torch.equal(m.policy.state_dict()[key], value) for m in models)
            if 'value' in key:
                assert torch.equal(models[0].policy.state_dict()[key], models[1].policy.state_dict()[key])
        assert any(not torch.equal(models[0].policy.state_dict()[k], v)
                   for k, v in initial.policy.state_dict().items() if 'value' in k)
    finally:
        env.close()


@pytest.mark.parametrize('variant', ['grpo', 'dr_grpo', 'dapo_clip_loss', 'rloo'])
def test_mixed_group_updates_actor_not_critic(variant):
    import copy
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'))
    from jepa_rl_doom import PROTOCOL, ROOT, update
    from stable_baselines3 import PPO
    p = json.loads(PROTOCOL.read_text())
    model = PPO.load(ROOT/p['initial_run']/'ppo_history-0.zip', device='cpu')
    reference = copy.deepcopy(model.policy).eval()
    reference.requires_grad_(False)
    trajectories = []
    for length in [3, 5]:
        states = torch.zeros((length, 18))
        states[:, 0] = 1
        with torch.no_grad():
            dist = model.policy.get_distribution(states).distribution
            actions = dist.sample()
            logs = dist.log_prob(actions)
        trajectories.append({'states': states.numpy(), 'actions': actions.numpy(), 'old_log_probs': logs.numpy()})
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.0001)
    receipt = update(model, reference, optimizer, trajectories, [0., 1.], variant, p)
    assert receipt['updated'] and receipt['epochs'] == (1 if variant == 'rloo' else 4)
    before = reference.state_dict()
    after = model.policy.state_dict()
    assert any(not torch.equal(after[k], v) for k, v in before.items() if k.startswith('action_net.'))
    assert all(torch.equal(after[k], v) for k, v in before.items() if 'value' in k)
