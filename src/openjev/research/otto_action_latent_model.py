"""Small causal action-latent predictors and two explicit matched controls.

This is a prospective Torch component, not a trained world model. It loads no
data or state files and performs no collection, fitting or phase admission.
All three models share the same 31-to-28 observation GRUCell and output heads.
The prefix contains already observed public features, including past actions.

The blind API accepts only that prefix and proposed actions. Recurrent F maps
the current latent and one action to a prior. The action-blind control supplies
zero action inputs to the same F. The direct control predicts each horizon
from the initial prefix latent and a position-conditioned summary of only that
action prefix; it never feeds a predicted latent into its next blind forecast.
Its action-position encoder has shared weights, not untrained horizon-specific
slots. Pooling is order-sensitive but is not claimed to be lossless.

Normal-observation forecasts use the prior BEFORE assimilating that horizon's
features. G then updates the prior with those features for the next forecast.
Observed found flags are used only after the corresponding forecast: terminal
features are ignored and the carried latent is frozen thereafter. Logits remain
finite learned outputs; no predicted or observed label forces a probability.
Class order is odor 0,1,2,3,found. Absorbing target construction and loss masking
remain caller responsibilities. Costs are four centered normalized contrasts;
auxiliary outputs predict feature columns19:21, never supply future features.

All operations are CPU float32, with maximum rollout horizon8. Construction
preserves the caller's CPU RNG. Work counters report executed calls and rows,
not FLOPs or equal training/inference cost. No hidden state persists in a model.
"""
from __future__ import annotations

import math

import torch
from torch import nn

VERSION = "otto-action-latent-model-v1"
KINDS = ("action_recurrent", "action_blind", "direct_horizon")
FEATURE_DIM, WIDTH, MAX_HORIZON, MAX_PREFIX = 31, 28, 8, 2188
DIRECT_WIDTH = 48
PARAMETER_COUNTS = {"action_recurrent": 8299, "action_blind": 8299, "direct_horizon": 8107}
OUTPUT_FIELDS = {"outcome_logits", "cost_contrasts", "aux_features", "prior_states",
                 "posterior_states", "prefix_state", "work"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.layout == torch.strided and value.dtype == dtype and tuple(value.shape) == tuple(shape),
            "CPU " + name + " with exact dtype and shape")


def _actions(actions, batch=None):
    require(isinstance(actions, torch.Tensor) and actions.ndim == 2, "action matrix [B,H]")
    b, horizon = actions.shape
    require(b > 0 and 1 <= horizon <= MAX_HORIZON and (batch is None or batch == b),
            "nonempty matching action batch and horizon1..8")
    _tensor(actions, torch.int64, (b, horizon), "actions")
    require(bool(((actions >= 0) & (actions < 4)).all()), "four proposed action IDs")
    return b, horizon


def action_position_features(actions):
    """Fixed float32[B,H,9] tokens: onehot4, two sin/cos pairs, position/8.

    No trainable position table exists. Every positional coordinate is active
    during horizons1..4 as well as5..8. A caller requiring a causal prefix must
    select that prefix; this function returns the supplied tokens individually.
    """
    batch, horizon = _actions(actions)
    position = torch.arange(1, horizon + 1, dtype=torch.float32, device="cpu") / MAX_HORIZON
    angle = math.pi * position
    positional = torch.stack((angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos(), position), -1)
    onehot = torch.nn.functional.one_hot(actions, 4).to(torch.float32)
    return torch.cat((onehot, positional[None].expand(batch, -1, -1)), dim=-1)


class ActionLatentModel(nn.Module):
    """No persistent carry; every invocation starts from its supplied prefix."""

    def __init__(self, kind, seed=0, *, cost_scale=1.):
        require(type(kind) is str and kind in KINDS, "declared action-latent control")
        require(type(seed) is int and 0 <= seed < 2**32, "uint32 initialization seed")
        require(type(cost_scale) in (int, float), "real fixed cost scale")
        try:
            scale = float(cost_scale)
        except (ValueError, OverflowError) as error:
            raise ValueError("finite positive float32 cost scale") from error
        require(math.isfinite(scale) and 0 < scale <= torch.finfo(torch.float32).max,
                "finite positive float32 cost scale")
        super().__init__()
        self.kind, self.seed = kind, seed
        self.register_buffer("cost_scale", torch.tensor(scale, dtype=torch.float32, device="cpu"))
        require(bool(self.cost_scale > 0), "nonzero represented cost scale")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            # Common modules precede the different transitions, so equal seeds
            # also give identical initial observation encoders and output heads.
            self.assimilation = nn.GRUCell(FEATURE_DIM, WIDTH, device="cpu", dtype=torch.float32)
            self.outcome_head = nn.Linear(WIDTH, 5, device="cpu", dtype=torch.float32)
            self.cost_head = nn.Linear(WIDTH, 4, device="cpu", dtype=torch.float32)
            self.aux_head = nn.Linear(WIDTH, 2, device="cpu", dtype=torch.float32)
            # The head learns standardized residual coordinates. Its fixed
            # TRAIN-derived scale converts outputs back to normalized /64 units.
            with torch.no_grad():
                self.cost_head.weight.zero_()
                self.cost_head.bias.zero_()
            if kind == "direct_horizon":
                self.action_encoder = nn.Linear(9, DIRECT_WIDTH, device="cpu", dtype=torch.float32)
                self.direct_prior = nn.Linear(WIDTH + DIRECT_WIDTH + 1, WIDTH, device="cpu", dtype=torch.float32)
                self.transition = None
            else:
                self.transition = nn.GRUCell(4, WIDTH, device="cpu", dtype=torch.float32)
                self.action_encoder = self.direct_prior = None
        require(sum(p.numel() for p in self.parameters()) == PARAMETER_COUNTS[kind], "exact control parameter count")

    def parameter_metadata(self):
        named = list(self.named_parameters())
        return {"kind": self.kind, "hidden_dim": WIDTH, "cost_scale": float(self.cost_scale),
                "count": sum(p.numel() for _, p in named),
                "trainable_count": sum(p.numel() for _, p in named if p.requires_grad),
                "parameters": {name: {"shape": list(p.shape), "count": p.numel()} for name, p in named},
                "zero_proposed_action_weight_parameters": 336 if self.kind == "action_blind" else 0,
                "direct_descriptor": "shared position-conditioned token MLP, causal mean pool" if self.kind == "direct_horizon" else None}

    def _parameters_valid(self):
        require(self.kind in KINDS and sum(p.numel() for p in self.parameters()) == PARAMETER_COUNTS[self.kind],
                "unchanged declared model dimensions")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32 and bool(torch.isfinite(p).all())
                    for p in self.parameters()), "finite CPU float32 model parameters")
        _tensor(self.cost_scale, torch.float32, (), "fixed cost scale buffer")
        require(bool(torch.isfinite(self.cost_scale)) and bool(self.cost_scale > 0), "positive fixed cost scale")

    def _inputs(self, prefix, prefix_lengths, actions):
        self._parameters_valid()
        require(isinstance(prefix, torch.Tensor) and prefix.ndim == 3, "observed prefix [B,T,31]")
        batch, span, dimension = prefix.shape
        require(batch > 0 and 1 <= span <= MAX_PREFIX and dimension == FEATURE_DIM, "bounded observed prefix")
        _tensor(prefix, torch.float32, (batch, span, FEATURE_DIM), "prefix")
        _tensor(prefix_lengths, torch.int64, (batch,), "prefix lengths")
        require(bool(((prefix_lengths >= 1) & (prefix_lengths <= span)).all()), "nonempty active prefix lengths")
        active = torch.arange(span, device="cpu")[None] < prefix_lengths[:, None]
        require(bool(torch.isfinite(prefix[active]).all()), "finite active observed prefix")
        _, horizon = _actions(actions, batch)
        return batch, horizon

    @staticmethod
    def _work():
        return {name: 0 for name in ("prefix_assimilation_calls", "prefix_assimilation_rows",
                "transition_calls", "transition_rows", "direct_token_calls", "direct_token_rows",
                "direct_prior_calls", "direct_prior_rows", "observation_assimilation_calls",
                "observation_assimilation_rows", "outcome_head_calls", "cost_head_calls",
                "aux_head_calls", "readout_rows", "terminal_frozen_rows")}

    def _encode(self, prefix, lengths, work):
        hidden = prefix.new_zeros((len(prefix), WIDTH))
        for step in range(prefix.shape[1]):
            lanes = torch.nonzero(lengths > step, as_tuple=False).flatten()
            if not lanes.numel():
                break
            value = self.assimilation(prefix[lanes, step], hidden[lanes])
            hidden = hidden.index_copy(0, lanes, value)
            work["prefix_assimilation_calls"] += 1
            work["prefix_assimilation_rows"] += len(lanes)
        return hidden

    def _descriptor(self, actions, work):
        tokens = action_position_features(actions)
        embedded = self.action_encoder(tokens).tanh()
        work["direct_token_calls"] += 1
        work["direct_token_rows"] += actions.numel()
        horizon = embedded.new_full((len(actions), 1), actions.shape[1] / MAX_HORIZON)
        return torch.cat((embedded.mean(dim=1), horizon), dim=1)

    def direct_descriptor(self, actions):
        """Learned float32[B,H,49] causal prefix descriptors, never fixed features.

        We re-encode each prefix at its own shape, avoiding both future actions
        and horizon-dependent GEMM geometry in an earlier descriptor. This
        deliberately costs H(H+1)/2 token rows per example for a blind block.
        """
        self._parameters_valid()
        require(self.kind == "direct_horizon", "direct descriptor only for direct control")
        _, horizon = _actions(actions)
        work = self._work()
        return torch.stack([self._descriptor(actions[:, :step + 1], work) for step in range(horizon)], dim=1)

    def _prior(self, hidden, actions, work):
        if self.kind == "direct_horizon":
            descriptor = self._descriptor(actions, work)
            prior = self.direct_prior(torch.cat((hidden, descriptor), dim=1)).tanh()
            work["direct_prior_calls"] += 1
            work["direct_prior_rows"] += len(hidden)
        else:
            action = torch.nn.functional.one_hot(actions[:, -1], 4).to(torch.float32)
            if self.kind == "action_blind":
                action = torch.zeros_like(action)
            prior = self.transition(action, hidden)
            work["transition_calls"] += 1
            work["transition_rows"] += len(hidden)
        return prior

    def _read(self, prior, work):
        logits = self.outcome_head(prior)
        raw = self.cost_head(prior)
        costs = self.cost_scale * (raw - raw.mean(dim=-1, keepdim=True))
        auxiliary = self.aux_head(prior)
        for name in ("outcome_head_calls", "cost_head_calls", "aux_head_calls"):
            work[name] += 1
        work["readout_rows"] += len(prior)
        return logits, costs, auxiliary

    @staticmethod
    def _result(prefix, priors, posteriors, outputs, work):
        result = {"outcome_logits": torch.stack([x[0] for x in outputs], 1),
                  "cost_contrasts": torch.stack([x[1] for x in outputs], 1),
                  "aux_features": torch.stack([x[2] for x in outputs], 1),
                  "prior_states": torch.stack(priors, 1),
                  "posterior_states": None if posteriors is None else torch.stack(posteriors, 1),
                  "prefix_state": prefix, "work": work}
        require(all(value is None or isinstance(value, dict) or bool(torch.isfinite(value).all())
                    for value in result.values()), "finite action-latent outputs")
        return result

    def blind_rollout(self, prefix, prefix_lengths, actions):
        """Forecast horizons1..H with no continuation observations or labels."""
        _, horizon = self._inputs(prefix, prefix_lengths, actions)
        work = self._work()
        initial = self._encode(prefix, prefix_lengths, work)
        hidden, priors, outputs = initial, [], []
        for step in range(horizon):
            # The direct arm always anchors at the original observed prefix.
            anchor = initial if self.kind == "direct_horizon" else hidden
            selected = actions[:, :step + 1] if self.kind == "direct_horizon" else actions[:, step:step + 1]
            hidden = self._prior(anchor, selected, work)
            priors.append(hidden)
            outputs.append(self._read(hidden, work))
        return self._result(initial, priors, None, outputs, work)

    def forward(self, prefix, prefix_lengths, actions):
        return self.blind_rollout(prefix, prefix_lengths, actions)

    def normal_rollout(self, prefix, prefix_lengths, actions, continuation, *, found):
        """One-step prior forecasts, then observed assimilation for the next step.

        continuation[B,H,31] at indexh is the observation AFTER actionh. Its
        odor, refreshed belief and other features cannot affect predictionh.
        found[B,H] records observed found events/status; the first true entry
        freezes that lane after forecasting it. Terminal/past-terminal features
        are unconsumed and may contain poison. Future actions still use IDs0..3.
        Direct prediction uses a one-action block after each new observation.
        """
        batch, horizon = self._inputs(prefix, prefix_lengths, actions)
        _tensor(continuation, torch.float32, (batch, horizon, FEATURE_DIM), "observed continuation")
        _tensor(found, torch.bool, (batch, horizon), "observed found flags")
        observed_active = found.to(torch.int64).cumsum(dim=1) == 0
        require(bool(torch.isfinite(continuation[observed_active]).all()), "finite consumed continuation observations")
        work = self._work()
        initial = self._encode(prefix, prefix_lengths, work)
        hidden = initial
        terminated = torch.zeros(batch, dtype=torch.bool, device="cpu")
        priors, posteriors, outputs = [], [], []
        for step in range(horizon):
            alive = torch.nonzero(~terminated, as_tuple=False).flatten()
            prior = hidden.clone()
            if alive.numel():
                value = self._prior(hidden[alive], actions[alive, step:step + 1], work)
                prior = prior.index_copy(0, alive, value)
            work["terminal_frozen_rows"] += int(terminated.sum())
            priors.append(prior)
            outputs.append(self._read(prior, work))
            assimilate = torch.nonzero(~terminated & ~found[:, step], as_tuple=False).flatten()
            posterior = prior.clone()
            if assimilate.numel():
                value = self.assimilation(continuation[assimilate, step], prior[assimilate])
                posterior = posterior.index_copy(0, assimilate, value)
                work["observation_assimilation_calls"] += 1
                work["observation_assimilation_rows"] += len(assimilate)
            posteriors.append(posterior)
            hidden = posterior
            terminated = terminated | found[:, step]
        return self._result(initial, priors, posteriors, outputs, work)


def make_model(kind, seed=0, *, cost_scale=1.):
    return ActionLatentModel(kind, seed, cost_scale=cost_scale)
