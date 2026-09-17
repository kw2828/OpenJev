"""Small structured-observation adaptation of SRPO, not a V-JEPA replication.

Reward convention follows sii-research/siiRL at 89d8764b6133ea78557a22eef32c68ad92946d45.
Implemented independently: DBSCAN(min_samples=2), decreasing sigmoid, group advantages.
"""
import numpy as np
import torch
from torch import nn


def success_centers(embeddings, eps=.5):
    """DBSCAN for min_samples=2 (including self), in standardized success space.

With min_samples=2, every point with a neighbor is core. Connected components
of the radius graph are therefore the clusters; isolated points are noise.
Like the pinned reference, use the overall mean when all points are noise.
"""
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite embedding matrix')
    scale = x.std(axis=0)
    z = (x-x.mean(axis=0))/np.where(scale > 1e-12, scale, 1.)
    neighbors = np.linalg.norm(z[:, None]-z[None, :], axis=2) <= eps
    visited, centers = set(), []
    for i in range(len(x)):
        if i in visited or neighbors[i].sum() < 2:
            continue
        members, pending = set(), [i]
        while pending:
            j = pending.pop()
            if j in members:
                continue
            members.add(j)
            pending.extend(k for k in np.flatnonzero(neighbors[j]) if k not in members)
        visited.update(members)
        centers.append(x[sorted(members)].mean(axis=0))
    return np.asarray(centers) if centers else x.mean(axis=0, keepdims=True)


def reference_rewards(embeddings, successful, weight=.6):
    """One group only. No history bank, labels from other tasks, or eval references."""
    x, good = np.asarray(embeddings, dtype=float), np.asarray(successful, dtype=bool)
    if x.ndim != 2 or good.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid trajectory embeddings or success labels')
    if not 0 < weight < 1:
        raise ValueError('Failure credit must be smaller than success credit')
    rewards = good.astype(float)
    if good.any() and (~good).any():
        centers = success_centers(x[good])
        d = np.linalg.norm(x[~good, None]-centers[None, :], axis=2).min(axis=1)
        width = np.ptp(d)
        normalized = (d-d.min())/width if width >= 1e-6 else np.full_like(d, .5)
        rewards[~good] = weight/(1+np.exp(-10*(.5-normalized)))
    return rewards


def group_advantages(rewards):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.isfinite(rewards).all():
        raise ValueError('Expected at least two finite group rewards')
    return (rewards-rewards.mean())/(rewards.std(ddof=1)+1e-6)


class DynamicsEncoder(nn.Module):
    """One-step observation predictor, trained only on historical training traces.

This is a small learned state representation, not a pretrained video world model.
It receives six current observation features; action enters only the decoder.
"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 8), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 6))

    def forward(self, observations, actions):
        return self.decoder(torch.cat((self.encoder(observations), actions[:, None]), dim=1))

    @torch.no_grad()
    def trajectory(self, observations):
        z = self.encoder(torch.as_tensor(observations, dtype=torch.float32)).numpy()
        return np.r_[z.mean(axis=0), z[-1]]


def raw_trajectory(observations):
    x = np.asarray(observations, dtype=float)
    return np.r_[x.mean(axis=0), x[-1]]


def policy_loss(policy, states, actions, old_log_probs, ref_probs, advantages, lengths,
                clip=.2, kl_weight=.01, entropy_weight=.01):
    """Exact categorical KL penalty, clipped ratios, equal trajectory weighting.

Only actual transitions are concatenated. No padded/terminal steps contribute.
The critic is not called. A positive KL is a penalty in a minimized loss.
"""
    dist = policy.get_distribution(states).distribution
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(torch.clamp(log_probs-old_log_probs, -20., 20.))
    surrogate = torch.minimum(ratio*advantages, ratio.clamp(1-clip, 1+clip)*advantages)
    kl = (dist.probs*(dist.logits-torch.log(ref_probs.clamp_min(1e-8)))).sum(dim=1)
    losses = -surrogate + kl_weight*kl - entropy_weight*dist.entropy()
    loss = torch.stack([part.mean() for part in torch.split(losses, lengths)]).mean()
    return loss, {'kl': float(kl.detach().mean()),
                  'clip_fraction': float(((ratio.detach()-1).abs() > clip).float().mean())}
