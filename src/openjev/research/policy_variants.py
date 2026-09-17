"""Explicit GRPO loss variants and a categorical REINFORCE leave-one-out baseline."""
import numpy as np
import torch


def advantages(rewards, variant):
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError('Expected at least two finite rewards')
    centered = r-r.mean()
    if variant in ('grpo', 'dapo_clip_loss'):
        return centered/(r.std(ddof=1)+1e-6)
    if variant == 'dr_grpo':
        return centered
    if variant == 'rloo':
        return centered*len(r)/(len(r)-1)
    raise ValueError('Unknown advantage variant')


def reduce_steps(values, lengths, variant, horizon):
    if sum(lengths) != len(values) or min(lengths) <= 0 or max(lengths) > horizon:
        raise ValueError('Invalid trajectory lengths')
    if variant == 'grpo':
        return torch.stack([x.mean() for x in torch.split(values, lengths)]).mean()
    if variant == 'dapo_clip_loss':
        return values.mean()
    if variant in ('dr_grpo', 'rloo'):
        return values.sum()/(len(lengths)*horizon)
    raise ValueError('Unknown loss variant')


def actor_loss(policy, states, actions, old_logs, reference_probs, adv, lengths, variant,
               horizon=180, kl_weight=.01, entropy_weight=.01):
    dist = policy.get_distribution(states).distribution
    logs = dist.log_prob(actions)
    ratio = torch.exp((logs-old_logs).clamp(-20, 20))
    if variant == 'rloo':
        # One on-policy REINFORCE step only, no PPO ratio or clipping.
        gain = logs*adv
    else:
        upper = 1.28 if variant == 'dapo_clip_loss' else 1.2
        gain = torch.minimum(ratio*adv, ratio.clamp(.8, upper)*adv)
    kl = (dist.probs*(dist.logits-reference_probs.clamp_min(1e-8).log())).sum(-1)
    terms = -gain+kl_weight*kl-entropy_weight*dist.entropy()
    return reduce_steps(terms, lengths, variant, horizon), {
        'kl': float(kl.detach().mean()), 'entropy': float(dist.entropy().detach().mean())}
