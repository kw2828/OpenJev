"""Exact public-history filtering over a finite observation-noise schedule bank.

The schedule index is a persistent hidden variable. Its joint posterior with
the eight physical states is retained; factorizing those two marginals would
lose correlations. The caller supplies a prospective bank and prior, not a
realized private schedule or target state. Emission changes at the specified
event index, before that event. Reset has a uniform physical-state prior and
no transition or found hazard. First found is scored and then absorbing.

The supplied law and schedule family are structural prior knowledge. Exactness
is relative to those assumptions; it is not a claim that arbitrary learned
fields equal the physical world. No RNG, model framework or file I/O is used.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

VERSION = 'finite-schedule-filter-v1'
TOLERANCE = 1e-12
FORK_HORIZON = 8
FIELD_SHAPES = MappingProxyType({'A': (4, 8, 8), 'found': (4, 8),
                               'emission': (4, 8), 'costs': (4, 8)})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, dtype, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
            and value.shape == shape and np.isfinite(value).all(), 'finite exact array schema: ' + name)


def unit(value, axis, name):
    require(np.isfinite(value).all() and (value >= 0).all()
            and (np.abs(value.sum(axis=axis, dtype=np.float64) - 1) <= TOLERANCE).all(),
            'nonnegative unit probability law: ' + name)


def owned(value):
    result = np.array(value, copy=True, order='C')
    result.setflags(write=False)
    return result


def schedule_bank(steps=32):
    """Return owned float64 [36,steps+1] schedules and [36] hypothesis prior.

    Rows0/1 are static .12/.30. Rows2..18 switch .12 to .30 at event8..24;
    rows19..35 reverse that direction. Each of the four families has prior1/4.
    Static two-hypothesis control: use bank[:2] and renormalize prior[:2].
    """
    require(type(steps) is int and 24 <= steps < 2**32, 'integer steps at least24')
    epsilons = np.empty((36, steps + 1), dtype=np.float64)
    epsilons[0], epsilons[1] = .12, .30
    for direction, (before, after) in enumerate(((.12, .30), (.30, .12))):
        for offset, switch in enumerate(range(8, 25)):
            row = 2 + 17 * direction + offset
            epsilons[row, :switch], epsilons[row, switch:] = before, after
    prior = np.full(36, .25 / 17, dtype=np.float64)
    prior[:2] = .25
    unit(prior, 0, 'schedule prior')
    return epsilons, prior


class ScheduleFilter:
    """Owned fixed law with exact Bayesian joint[schedule,state] recursion.

    General banks may have any positive hypothesis count and any nonnegative
    action horizon, permitting short independent fixtures. No realized noise
    path can be supplied to filter(). All public arrays include reset and the
    full absorbing label4 suffix. Outputs are fresh owned float64 arrays.
    """

    def __init__(self, fields, epsilons, prior):
        require(isinstance(fields, dict) and set(fields) == set(FIELD_SHAPES), 'exact A/found/emission/costs fields')
        for name, shape in FIELD_SHAPES.items():
            array(fields[name], 'float64', shape, name)
        require((fields['A'] >= 0).all() and ((fields['found'] >= 0) & (fields['found'] <= 1)).all(), 'nonnegative surviving and found laws')
        total = fields['A'].sum(axis=1, dtype=np.float64) + fields['found']
        require((np.abs(total - 1) <= TOLERANCE).all(), 'A columns plus found equal one')
        unit(fields['emission'], 0, 'base emission')
        require(isinstance(epsilons, np.ndarray) and epsilons.ndim == 2
                and epsilons.shape[0] > 0 and epsilons.shape[1] > 0, 'nonempty schedule matrix')
        array(epsilons, 'float64', epsilons.shape, 'schedule epsilon')
        require(((epsilons >= .12) & (epsilons <= .75)).all(), 'epsilon within [.12,.75]')
        array(prior, 'float64', (epsilons.shape[0],), 'schedule prior')
        unit(prior, 0, 'schedule prior')
        self.steps, self.hypotheses = epsilons.shape[1] - 1, epsilons.shape[0]
        self._fields = MappingProxyType({name: owned(value) for name, value in fields.items()})
        self._epsilons, self._prior = owned(epsilons), owned(prior)
        q = (self._epsilons - .12) / (.75 - .12)
        emissions = ((1 - q[:, :, None, None]) * self._fields['emission'][None, None]
                     + q[:, :, None, None] / 4)
        unit(emissions, 2, 'all prospective schedule emissions')
        self._emissions = owned(emissions)

    def filter(self, actions, observations):
        require(isinstance(actions, np.ndarray) and actions.ndim == 2, 'public action matrix')
        batch = actions.shape[0]
        array(actions, 'int64', (batch, self.steps), 'actions')
        array(observations, 'int64', (batch, self.steps + 1), 'observations')
        require(((actions >= 0) & (actions < 4)).all() and ((observations >= 0) & (observations <= 4)).all(), 'public action and event domains')
        require((observations[:, 0] < 4).all(), 'reset is an ordinary odor')
        found = observations == 4
        require(np.array_equal(found, np.maximum.accumulate(found, axis=1)), 'found has an absorbing suffix')
        probabilities = np.zeros((batch, self.steps + 1, 5), dtype=np.float64)
        states = np.zeros((batch, self.steps + 1, 8), dtype=np.float64)
        schedules = np.zeros((batch, self.steps + 1, self.hypotheses), dtype=np.float64)
        joint = np.broadcast_to(self._prior[None, :, None] / 8, (batch, self.hypotheses, 8)).copy()
        absorbed = np.zeros(batch, dtype=bool)
        for t in range(self.steps + 1):
            probabilities[:, t, 4] = 1
            active = np.flatnonzero(~absorbed)
            if active.size:
                before = joint[active]
                if t == 0:
                    predicted = before
                    found_probability = np.zeros(len(active), dtype=np.float64)
                else:
                    action = actions[active, t - 1]
                    predicted = np.einsum('bij,bhj->bhi', self._fields['A'][action], before)
                    found_probability = np.einsum('bi,bi->b', before.sum(1), self._fields['found'][action])
                law = self._emissions[:, t]
                probability = np.column_stack((np.einsum('bhi,hoi->bo', predicted, law), found_probability))
                unit(probability, 1, 'prequential event prediction')
                probabilities[active, t] = probability
                observed = observations[active, t]
                evidence = probability[np.arange(len(active)), observed]
                require(np.isfinite(evidence).all() and (evidence > 0).all(), 'strictly positive observed event evidence')
                updated = np.zeros_like(before)
                ordinary = np.flatnonzero(observed < 4)
                if ordinary.size:
                    likelihood = law[:, observed[ordinary], :].transpose(1, 0, 2)
                    updated[ordinary] = predicted[ordinary] * likelihood / evidence[ordinary, None, None]
                    unit(updated[ordinary].reshape(len(ordinary), -1), 1, 'posterior schedule-state joint')
                joint[active] = updated
                absorbed[active[observed == 4]] = True
            states[:, t] = joint.sum(axis=1, dtype=np.float64)
            schedules[:, t] = joint.sum(axis=2, dtype=np.float64)
        costs = states @ self._fields['costs'].T
        require(np.isfinite(costs).all(), 'finite posterior expected costs')
        return {'probabilities': probabilities, 'post_states': states,
                'post_costs': costs, 'schedule_posterior': schedules}

    def fork(self, states, fork_actions):
        """Eight blind steps from owned caller states; no future labels/noise.

        states are finite nonnegative mass at public boundaries, with total at
        most one. Zero mass is absorbing. costs are unconditional C@A...A@state;
        choose an action only after integrating these expected cost vectors.
        """
        require(isinstance(states, np.ndarray) and states.ndim == 3, 'boundary state tensor')
        batch = states.shape[0]
        array(states, 'float64', (batch, self.steps + 1, 8), 'boundary states')
        require((states >= 0).all() and (states.sum(2) <= 1 + TOLERANCE).all(), 'bounded boundary mass')
        array(fork_actions, 'int64', (batch, self.steps + 1, FORK_HORIZON), 'committed fork actions')
        require(((fork_actions >= 0) & (fork_actions < 4)).all(), 'fork action domain')
        current = states.copy()
        costs = np.zeros((batch, self.steps + 1, FORK_HORIZON, 4), dtype=np.float64)
        survival = np.zeros((batch, self.steps + 1, FORK_HORIZON), dtype=np.float64)
        for h in range(FORK_HORIZON):
            current = np.einsum('btij,btj->bti', self._fields['A'][fork_actions[:, :, h]], current)
            mass = current.sum(axis=2, dtype=np.float64)
            require(np.isfinite(current).all() and (current >= 0).all()
                    and (mass <= 1 + TOLERANCE).all(), 'finite surviving blind mass')
            costs[:, :, h] = current @ self._fields['costs'].T
            survival[:, :, h] = mass
        require(np.isfinite(costs).all(), 'finite unconditional blind costs')
        return {'costs': costs, 'survival': survival}
