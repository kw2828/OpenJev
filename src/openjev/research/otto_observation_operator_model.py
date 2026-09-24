"""Prospective positive observation operators with linear decision moments.

No fitting, files, simulator, teacher or persistent carry lives here. The
observed operator B[a,o,next,current] has nonnegative entries. A column softmax
jointly allocates mass to all four odors and their next coordinates plus one
found outcome. Tied blind propagation sums the four B matrices. The untied
control uses its own column-softmax substochastic transition and found row.
Observed forecasts/conditioning use B in BOTH kinds, isolating the blind tie.

Observed nonterminal state is normalized conditional mass. Blind state is
unnormalized surviving mass. Never divide a blind forecast by survival. A
bias-free, centered linear cost readout commutes with branch summation. This
identity holds inside this model, not necessarily for the nonlinear teacher.
Observed found sets all mass to zero; later costs remain exactly zero.

Precision: CPU float32 GRU(31,28); float64 projection, probabilities, states,
operators and readout. Cost scale is a fixed float64 buffer. Mixed precision
and extra untied parameters do not imply matched speed/compute. Runtime guards
materialize O(batch*width**2) probability products to reject positive-product
underflow; reported structural work is calls/rows, not all validation FLOPs.
Softmax underflow, zero observed evidence, nonfinite arithmetic, negative mass
or material mass-conservation failure raise ValueError without state repair.
Only documented roundoff-sized excess above unit mass is tolerated, never
clipped. Observations are odor/found IDs, not the old GRU's full31-feature
continuation: a future experiment must match that interface explicitly.

Initial logits are .05 times standard normal. Near-flat columns initially
assign found about1/(4*width+1), not1/5. Dense mixing can rapidly remove prefix
distinctions; no information-retention or trained-model claim follows. Untied
raw blind logits use logsumexp of the SAME observed logits, so initial blind
functions agree up to float64 reduction/softmax rounding, not bitwise identity.
"""
from __future__ import annotations

import math

import torch
from torch import nn

VERSION = 'otto-observation-operator-v1'
KINDS = ('tied', 'untied')
FEATURE_DIM, ENCODER_WIDTH, MAX_PREFIX, MAX_HORIZON = 31, 28, 2188, 8
WORK_KEYS = ('prefix_assimilation_calls', 'prefix_assimilation_rows', 'projection_calls', 'projection_rows',
    'prefix_softmax_calls', 'operator_observed_softmax_calls', 'operator_blind_softmax_calls',
    'operator_marginal_sum_calls', 'blind_transition_calls', 'blind_transition_rows',
    'observed_branch_calls', 'observed_branch_rows', 'observed_conditioning_rows',
    'cost_readout_calls', 'cost_readout_rows', 'absorbed_rows')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided
            and value.dtype == dtype and tuple(value.shape) == tuple(shape), 'CPU exact dtype/shape: ' + name)


def _finite(value, name):
    require(bool(torch.isfinite(value).all()), 'finite ' + name)


def _work():
    return dict.fromkeys(WORK_KEYS, 0)


class ObservationOperatorModel(nn.Module):
    """Pure call-owned recurrent mass, no latent state stored between calls."""

    def __init__(self, kind='tied', seed=0, *, width=14, cost_scale=1.):
        require(type(kind) is str and kind in KINDS, 'declared operator kind')
        require(type(seed) is int and 0 <= seed < 2**32, 'uint32 initialization seed')
        require(type(width) is int and width > 0, 'positive latent width')
        require(type(cost_scale) in (int, float) and math.isfinite(cost_scale) and cost_scale > 0,
                'finite positive fixed cost scale')
        super().__init__()
        self.kind, self.seed, self.width = kind, seed, width
        self.register_buffer('cost_scale', torch.tensor(float(cost_scale), dtype=torch.float64, device='cpu'))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.assimilation = nn.GRUCell(FEATURE_DIM, ENCODER_WIDTH, device='cpu', dtype=torch.float32)
            self.projection = nn.Linear(ENCODER_WIDTH, width, device='cpu', dtype=torch.float64)
            self.readout = nn.Linear(width, 4, bias=False, device='cpu', dtype=torch.float64)
            self.observed_logits = nn.Parameter(.05 * torch.randn(4, 4 * width + 1, width, dtype=torch.float64, device='cpu'))
            if kind == 'untied':
                observed = self.observed_logits.detach()
                aggregated = torch.logsumexp(observed[:, :-1].reshape(4, 4, width, width), dim=1)
                self.blind_logits = nn.Parameter(torch.cat((aggregated, observed[:, -1:, :]), dim=1))
            else:
                self.register_parameter('blind_logits', None)
        self._parameters_valid()

    def _tolerance(self):
        """Explicit conservative float64 mass-check allowance, no repair."""
        return 64 * max(self.width, 4) * torch.finfo(torch.float64).eps

    def _parameters_valid(self):
        require(self.kind in KINDS and type(self.width) is int and self.width > 0, 'unchanged model configuration')
        w = self.width
        shapes = {'assimilation.weight_ih': (84, 31), 'assimilation.weight_hh': (84, 28),
            'assimilation.bias_ih': (84,), 'assimilation.bias_hh': (84,),
            'projection.weight': (w, 28), 'projection.bias': (w,), 'readout.weight': (4, w),
            'observed_logits': (4, 4 * w + 1, w)}
        if self.kind == 'untied':
            shapes['blind_logits'] = (4, w + 1, w)
        named = dict(self.named_parameters())
        require(set(named) == set(shapes), 'exact parameter roster')
        for name, value in named.items():
            dtype = torch.float32 if name.startswith('assimilation.') else torch.float64
            _tensor(value, dtype, shapes[name], name)
            _finite(value, name)
        _tensor(self.cost_scale, torch.float64, (), 'cost scale')
        require(bool(torch.isfinite(self.cost_scale)) and bool(self.cost_scale > 0), 'positive fixed scale')

    def parameter_metadata(self):
        self._parameters_valid()
        named = dict(self.named_parameters())
        groups = {'encode_prefix': [name for name in named if name.startswith(('assimilation.', 'projection.'))],
            'blind': [name for name in named if self.kind == 'tied' or name != 'observed_logits'],
            'observed': [name for name in named if name != 'blind_logits'], 'combined': list(named)}
        records = {name: {'shape': list(value.shape), 'count': value.numel(), 'bytes': value.numel() * value.element_size(),
            'dtype': str(value.dtype), 'requires_grad': value.requires_grad} for name, value in named.items()}
        return {'version': VERSION, 'kind': self.kind, 'seed': self.seed, 'encoder_width': 28, 'mass_width': self.width,
            'cost_scale': float(self.cost_scale), 'parameters': records, 'count': sum(p.numel() for p in named.values()),
            'trainable_count': sum(p.numel() for p in named.values() if p.requires_grad),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in named.values()), 'buffer_bytes': 8,
            'effective_parameter_names': groups,
            'effective_counts': {key: sum(named[name].numel() for name in names if named[name].requires_grad) for key, names in groups.items()},
            'precision': {'prefix_and_encoder': 'torch.float32', 'projection_operators_state_readout': 'torch.float64'},
            'mass_roundoff_allowance': self._tolerance(), 'max_horizon': MAX_HORIZON,
            'scope': 'Structural parameter/storage accounting; call/row work excludes validation FLOPs, autograd storage and temporary products. No matched compute claim.'}

    def _operators(self, work):
        self._parameters_valid()
        w = self.width
        probabilities = torch.softmax(self.observed_logits, dim=1)
        work['operator_observed_softmax_calls'] += 1
        _finite(probabilities, 'observation probabilities')
        require(bool((probabilities > 0).all()), 'positive softmax entries: probability underflow is rejected')
        observed = probabilities[:, :-1].reshape(4, 4, w, w)
        found = probabilities[:, -1]
        marginal = observed.sum(dim=1)
        work['operator_marginal_sum_calls'] += 1
        if self.kind == 'tied':
            blind, blind_found = marginal, found
        else:
            independent = torch.softmax(self.blind_logits, dim=1)
            work['operator_blind_softmax_calls'] += 1
            _finite(independent, 'blind probabilities')
            require(bool((independent > 0).all()), 'positive blind softmax entries: probability underflow is rejected')
            blind, blind_found = independent[:, :-1], independent[:, -1]
        for matrix, terminal in ((marginal, found), (blind, blind_found)):
            require(bool(((matrix.sum(-2) + terminal - 1).abs() <= self._tolerance()).all()), 'column-stochastic outgoing mass')
        return {'observed': observed, 'found': found, 'blind': blind, 'blind_found': blind_found}, marginal

    def operators(self):
        """Differentiable float64 probabilities; returned storage never aliases parameters."""
        return self._operators(_work())[0]

    def _prefix_inputs(self, prefix, lengths):
        self._parameters_valid()
        require(isinstance(prefix, torch.Tensor) and prefix.ndim == 3, 'prefix [B,T,31]')
        batch, span, features = prefix.shape
        require(batch > 0 and 1 <= span <= MAX_PREFIX and features == FEATURE_DIM, 'bounded nonempty prefix')
        _tensor(prefix, torch.float32, (batch, span, FEATURE_DIM), 'prefix')
        _tensor(lengths, torch.int64, (batch,), 'prefix lengths')
        require(bool(((lengths > 0) & (lengths <= span)).all()), 'positive active prefix lengths')
        active = torch.arange(span)[None] < lengths[:, None]
        _finite(prefix[active], 'consumed prefix')
        return batch

    def _encode(self, prefix, lengths, work):
        batch = self._prefix_inputs(prefix, lengths)
        hidden = prefix.new_zeros((batch, ENCODER_WIDTH))
        for step in range(prefix.shape[1]):
            lanes = torch.nonzero(lengths > step, as_tuple=False).flatten()
            if lanes.numel():
                value = self.assimilation(prefix[lanes, step], hidden[lanes])
                hidden = hidden.index_copy(0, lanes, value)
                work['prefix_assimilation_calls'] += 1
                work['prefix_assimilation_rows'] += len(lanes)
        logits = self.projection(hidden.to(torch.float64))
        work['projection_calls'] += 1
        work['projection_rows'] += batch
        _finite(logits, 'projected prefix')
        state = torch.softmax(logits, -1)
        work['prefix_softmax_calls'] += 1
        require(bool((state > 0).all()), 'positive prefix softmax: probability underflow is rejected')
        self._state(state, normalized=True)
        return state

    def encode_prefix(self, prefix, lengths):
        """Owned normalized float64 mass; ignores inactive padded rows."""
        return self._encode(prefix, lengths, _work())

    def _state(self, state, *, normalized=False):
        require(isinstance(state, torch.Tensor) and state.ndim == 2 and len(state) > 0, 'nonempty mass [B,width]')
        _tensor(state, torch.float64, (len(state), self.width), 'mass')
        _finite(state, 'mass')
        require(bool((state >= 0).all()), 'nonnegative mass')
        mass = state.sum(-1)
        require(bool((mass <= 1 + self._tolerance()).all()), 'subprobability mass')
        if normalized:
            require(bool(((mass == 0) | ((mass - 1).abs() <= self._tolerance())).all()), 'observed mass normalized or absorbed zero')
        return mass

    def _actions(self, actions, batch):
        _tensor(actions, torch.int64, (batch,), 'step actions')
        require(bool(((actions >= 0) & (actions < 4)).all()), 'four action IDs')

    @staticmethod
    def _multiply(matrix, state):
        products = matrix * state[:, None, :]
        require(not bool(((matrix > 0) & (state[:, None, :] > 0) & (products == 0)).any()),
                'positive probability product underflow is rejected')
        _finite(products, 'probability products')
        value = products.sum(-1)
        _finite(value, 'propagated mass')
        return value

    def _read(self, state, work):
        raw = self.readout(state)
        value = self.cost_scale * (raw - raw.mean(-1, keepdim=True))
        _finite(value, 'centered linear costs')
        work['cost_readout_calls'] += 1
        work['cost_readout_rows'] += len(state)
        return value

    def _prior(self, state, actions, matrix, terminal, work):
        mass = self._state(state)
        self._actions(actions, len(state))
        prior = self._multiply(matrix[actions], state)
        found = self._multiply(terminal[actions, None, :], state).squeeze(-1)
        survival = prior.sum(-1)
        require(bool(((survival + found - mass).abs() <= self._tolerance()).all()), 'conserved outgoing mass')
        require(bool((survival <= mass + self._tolerance()).all()), 'blind survival cannot materially increase')
        work['blind_transition_calls'] += 1
        work['blind_transition_rows'] += len(state)
        work['absorbed_rows'] += int((mass == 0).sum())
        return {'prior_state': prior, 'cost_contrasts': self._read(prior, work),
                'survival_mass': survival, 'found_increment': found}

    def blind_step(self, state, actions):
        """Unnormalized propagation; found_increment is newly lost mass only."""
        work = _work()
        operators, _ = self._operators(work)
        result = self._prior(state, actions, operators['blind'], operators['blind_found'], work)
        return {**result, 'work': work}

    def _observed(self, state, actions, observations, operators, marginal, work):
        mass = self._state(state, normalized=True)
        self._actions(actions, len(state))
        _tensor(observations, torch.int64, (len(state),), 'observations')
        require(bool(((observations >= 0) & (observations <= 4)).all()), 'odor/found observation IDs')
        absorbed = mass == 0
        require(bool((observations[absorbed] == 4).all()), 'absorbed zero state requires found observation')
        result = self._prior(state, actions, marginal, operators['found'], work)
        # The just-observed label cannot affect its preceding prediction.
        posterior = torch.zeros_like(state)
        evidence = torch.where(absorbed, torch.ones_like(mass), result['found_increment'])
        lanes = torch.nonzero(~absorbed & (observations < 4), as_tuple=False).flatten()
        if lanes.numel():
            branches = operators['observed'][actions[lanes], observations[lanes]]
            unnormalized = self._multiply(branches, state[lanes])
            probability = unnormalized.sum(-1)
            require(bool((probability > 0).all()), 'positive observed evidence; no normalization repair')
            conditioned = unnormalized / probability[:, None]
            self._state(conditioned, normalized=True)
            posterior = posterior.index_copy(0, lanes, conditioned)
            evidence = evidence.index_copy(0, lanes, probability)
            work['observed_branch_calls'] += 1
            work['observed_branch_rows'] += len(lanes)
            work['observed_conditioning_rows'] += len(lanes)
        require(bool((evidence > 0).all()), 'positive observed evidence including found')
        return {**result, 'posterior_state': posterior, 'evidence': evidence}

    def observed_step(self, state, actions, observations):
        """Predict with sum(B) first; normalize only the observed odor branch.

        Found observation maps to zero. An already absorbed lane has evidence1
        by convention, zero predictive costs and remains zero. This method does
        not use the untied blind operator for its predictive prior.
        """
        work = _work()
        operators, marginal = self._operators(work)
        return {**self._observed(state, actions, observations, operators, marginal, work), 'work': work}

    def _rollout_inputs(self, prefix, lengths, actions, observations=None):
        batch = self._prefix_inputs(prefix, lengths)
        require(isinstance(actions, torch.Tensor) and actions.ndim == 2, 'action block [B,H]')
        horizon = actions.shape[1]
        require(1 <= horizon <= MAX_HORIZON, 'horizon in1..8')
        _tensor(actions, torch.int64, (batch, horizon), 'action block')
        require(bool(((actions >= 0) & (actions < 4)).all()), 'four action IDs')
        if observations is not None:
            _tensor(observations, torch.int64, (batch, horizon), 'observation block')
            require(bool(((observations >= 0) & (observations <= 4)).all()), 'five observation IDs')
            found = observations == 4
            require(torch.equal(found, found.to(torch.int64).cummax(-1).values.bool()), 'absorbing found suffix required')
        return horizon

    def _rollout(self, prefix, lengths, actions, observations=None):
        horizon = self._rollout_inputs(prefix, lengths, actions, observations)
        work = _work()
        initial = self._encode(prefix, lengths, work)
        operators, marginal = self._operators(work)
        state, records = initial, []
        for step in range(horizon):
            if observations is None:
                result = self._prior(state, actions[:, step], operators['blind'], operators['blind_found'], work)
                state = result['prior_state']
            else:
                result = self._observed(state, actions[:, step], observations[:, step], operators, marginal, work)
                state = result['posterior_state']
            records.append(result)
        return {'prefix_state': initial, 'prior_states': torch.stack([r['prior_state'] for r in records], 1),
            'posterior_states': None if observations is None else torch.stack([r['posterior_state'] for r in records], 1),
            'cost_contrasts': torch.stack([r['cost_contrasts'] for r in records], 1),
            'survival_mass': torch.stack([r['survival_mass'] for r in records], 1),
            'found_increments': torch.stack([r['found_increment'] for r in records], 1),
            'evidence': None if observations is None else torch.stack([r['evidence'] for r in records], 1), 'work': work}

    def blind_rollout(self, prefix, lengths, actions):
        """survival_mass is cumulative; found_increments are newly lost mass."""
        return self._rollout(prefix, lengths, actions)

    def observed_rollout(self, prefix, lengths, actions, observations):
        """survival_mass is the one-step conditional mass before each observation.

        Every nonterminal posterior is normalized for the following step;
        already observed found has zero mass thereafter. No full continuation
        features or future observations enter the preceding cost prediction.
        found_increments are conditional one-step probabilities here: do not
        sum them across horizons as for unconditional blind lost mass.
        """
        require(observations is not None, 'explicit observed rollout labels')
        return self._rollout(prefix, lengths, actions, observations)

    def forward(self, prefix, lengths, actions):
        return self.blind_rollout(prefix, lengths, actions)


def make_model(kind='tied', seed=0, *, width=14, cost_scale=1.):
    return ObservationOperatorModel(kind, seed, width=width, cost_scale=cost_scale)
