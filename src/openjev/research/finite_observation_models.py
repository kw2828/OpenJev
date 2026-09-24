"""Prospective synthetic learning adapters; no world knowledge in these models."""
from __future__ import annotations

import torch
from torch import nn

from openjev.research.otto_observation_operator_model import make_model as operator_model

ARMS = ('tied_dense', 'untied_dense', 'tied_retentive', 'untied_retentive', 'gru')


def require(ok, message):
    if not ok:
        raise ValueError(message)


class OperatorAdapter(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.core = operator_model('untied' if arm.startswith('untied') else 'tied', seed, width=14)
        if arm.endswith('retentive'):
            width = self.core.width
            transition = .95 * torch.eye(width, dtype=torch.float64) + .05 / width
            found = 1. / (4 * width + 1)
            branches = (.25 * (1 - found) * transition).repeat(4, 1)
            template = torch.cat((branches, torch.full((1, width), found, dtype=torch.float64)), 0)
            with torch.no_grad():
                self.core.observed_logits.add_(template.log()[None])
                if self.core.blind_logits is not None:
                    raw = self.core.observed_logits
                    marginal = torch.logsumexp(raw[:, :-1].reshape(4, 4, width, width), 1)
                    self.core.blind_logits.copy_(torch.cat((marginal, raw[:, -1:]), 1))

    def blind_rollout(self, prefix, lengths, actions):
        return self.core.blind_rollout(prefix, lengths, actions)

    def observed_rollout(self, prefix, lengths, actions, observations):
        result = self.core.observed_rollout(prefix, lengths, actions, observations)
        previous = torch.cat((result['prefix_state'][:, None], result['posterior_states'][:, :-1]), 1)
        operators = self.core.operators()
        event_law = torch.cat((operators['observed'].sum(2), operators['found'][:, None]), 1)
        probabilities = torch.einsum('bhoi,bhi->bho', event_law[actions], previous)
        terminal = torch.zeros_like(probabilities)
        terminal[..., 4] = 1
        probabilities = torch.where((previous.sum(-1) == 0)[..., None], terminal, probabilities)
        require(bool(torch.isfinite(probabilities).all()) and bool((probabilities >= 0).all()),
                'finite nonnegative event probabilities')
        return {**result, 'probabilities': probabilities,
                'adapter_extra_work': {'operator_materializations': 1,
                                       'event_probability_rows': actions.numel()},
                'work_scope': 'work counts the core only; event-law construction and einsum are additional. Timed adapter calls include both.'}


class GRUAdapter(nn.Module):
    """Same public tokens, no oracle filtering or supplied continuation summaries.

    Event probabilities are conditional on the carried nonterminal state.
    Blind survival is accumulated multiplicatively. Observed found is absorbing.
    A post-action forecast precedes assimilation of the current observation.
    """
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.GRUCell(31, 28, device='cpu', dtype=torch.float32)
            self.action_update = nn.GRUCell(4, 28, device='cpu', dtype=torch.float32)
            self.observation_update = nn.GRUCell(5, 28, device='cpu', dtype=torch.float32)
            self.cost_head = nn.Linear(28, 4, device='cpu', dtype=torch.float32)
            self.event_head = nn.Linear(28, 5, device='cpu', dtype=torch.float32)

    def _rollout(self, prefix, lengths, actions, observations=None):
        require(prefix.dtype == torch.float32 and prefix.ndim == 3 and prefix.shape[2] == 31,
                'float32 public prefix')
        batch, span, _ = prefix.shape
        require(lengths.dtype == torch.int64 and tuple(lengths.shape) == (batch,)
                and bool(((lengths > 0) & (lengths <= span)).all()), 'valid active prefix lengths')
        require(actions.dtype == torch.int64 and actions.ndim == 2 and len(actions) == batch
                and 1 <= actions.shape[1] <= 8 and bool(((actions >= 0) & (actions < 4)).all()),
                'bounded action block')
        if observations is not None:
            require(observations.dtype == torch.int64 and observations.shape == actions.shape
                    and bool(((observations >= 0) & (observations <= 4)).all()), 'observation block')
            found = observations == 4
            require(torch.equal(found, found.to(torch.int64).cummax(-1).values.bool()), 'absorbing suffix')
        hidden = prefix.new_zeros((batch, 28))
        for time in range(span):
            lanes = torch.nonzero(lengths > time, as_tuple=False).flatten()
            if lanes.numel():
                require(bool(torch.isfinite(prefix[lanes, time]).all()), 'finite consumed public features')
                hidden = hidden.index_copy(0, lanes, self.encoder(prefix[lanes, time], hidden[lanes]))
        mass = prefix.new_ones(batch)
        costs, survivals, events = [], [], []
        for time in range(actions.shape[1]):
            action = torch.nn.functional.one_hot(actions[:, time], 4).to(prefix.dtype)
            prior = self.action_update(action, hidden)
            probability = self.event_head(prior).softmax(-1)
            require(bool((probability > 0).all()), 'event probability underflow rejected')
            terminal = torch.zeros_like(probability)
            terminal[:, 4] = 1
            probability = torch.where((mass == 0)[:, None], terminal, probability)
            survival = mass * probability[:, :4].sum(-1)
            raw_cost = self.cost_head(prior)
            cost = survival[:, None] * (raw_cost - raw_cost.mean(-1, keepdim=True))
            costs.append(cost.to(torch.float64))
            survivals.append(survival.to(torch.float64))
            events.append(probability.to(torch.float64))
            if observations is None:
                hidden, mass = prior, survival
            else:
                label = observations[:, time]
                encoded = torch.nn.functional.one_hot(label, 5).to(prefix.dtype)
                updated = self.observation_update(encoded, prior)
                live = (mass > 0) & (label < 4)
                hidden = torch.where(live[:, None], updated, torch.zeros_like(updated))
                mass = live.to(prefix.dtype)
        result = {'cost_contrasts': torch.stack(costs, 1), 'survival_mass': torch.stack(survivals, 1),
                  'probabilities': torch.stack(events, 1)}
        require(all(bool(torch.isfinite(value).all()) for value in result.values()), 'finite GRU predictions')
        return result

    def blind_rollout(self, prefix, lengths, actions):
        return self._rollout(prefix, lengths, actions)

    def observed_rollout(self, prefix, lengths, actions, observations):
        return self._rollout(prefix, lengths, actions, observations)


def make_model(arm, seed):
    require(arm in ARMS and type(seed) is int and 0 <= seed < 2**32, 'registered arm and seed')
    return GRUAdapter(seed) if arm == 'gru' else OperatorAdapter(arm, seed)


def metadata(model):
    rows = {name: {'count': p.numel(), 'dtype': str(p.dtype), 'bytes': p.numel() * p.element_size()}
            for name, p in model.named_parameters()}
    return {'parameters': rows, 'parameter_count': sum(r['count'] for r in rows.values()),
            'parameter_bytes': sum(r['bytes'] for r in rows.values()),
            'buffer_bytes': sum(b.numel() * b.element_size() for b in model.buffers()),
            'scope': 'Parameter storage only; activations, validation products and optimizer storage are additional. No matched compute claim.'}


def soft_cross_entropy(prediction, target):
    require(prediction.shape == target.shape and bool(torch.isfinite(prediction).all())
            and bool(torch.isfinite(target).all()) and bool((prediction >= 0).all())
            and bool((target >= 0).all()), 'finite probability target/prediction')
    require(not bool(((target > 0) & (prediction == 0)).any()), 'positive target cannot have zero predicted probability')
    require(bool(((prediction.sum(-1) - 1).abs() <= 1e-6).all())
            and bool(((target.sum(-1) - 1).abs() <= 1e-6).all()), 'unit probability rows within float32 allowance')
    log_probability = torch.where(prediction > 0, prediction, torch.ones_like(prediction)).log()
    return -(target * log_probability).sum(-1).mean()


def objective(blind, observed, targets):
    blind_cost = (blind['cost_contrasts'] - targets['blind_costs']).square().mean()
    observed_cost = (observed['cost_contrasts'] - targets['observed_costs']).square().mean()
    survival = .5 * ((blind['survival_mass'] - targets['blind_survival']).square().mean()
                     + (observed['survival_mass'] - targets['observed_survival']).square().mean())
    events = soft_cross_entropy(observed['probabilities'], targets['observed_probabilities'])
    total = blind_cost + observed_cost + survival + events
    require(bool(torch.isfinite(total)), 'finite combined objective')
    return total
