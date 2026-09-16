"""Bayesian outcome prediction for bounded actions, not proprietary Jev RLCD.

A Gaussian-prior logistic model uses a Laplace posterior approximation. Reports
P(observed success | features, executing action), never P(action is optimal).
"""

from dataclasses import dataclass

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -x))


def brier_reward(probability, outcome):
    """A standard proper-score baseline for fixed, verifiable Bernoulli events.

    Combining this with decision-dependent incentives does not by itself preserve
    honest reporting, calibration under shift, or trajectory safety.
    """
    if not np.isfinite(probability) or not 0 <= probability <= 1 or outcome not in (0, 1):
        raise ValueError("Require a probability in [0, 1] and a binary outcome")
    return -float((probability - outcome) ** 2)


@dataclass(frozen=True)
class OutcomeBelief:
    map_probability: float
    mean_probability: float
    epistemic_variance: float
    aleatoric_variance: float
    lower_probability: float
    upper_probability: float


class BayesianLogistic:
    def __init__(self, dimensions, prior_precision=1.0):
        if dimensions < 1 or not np.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("Positive dimensions and prior precision required")
        self.dimensions = dimensions
        self.prior_precision = float(prior_precision)
        self.mean = np.zeros(dimensions)
        self.covariance = np.eye(dimensions) / prior_precision
        self.nodes, self.weights = np.polynomial.hermite.hermgauss(20)
        self.weights /= np.sqrt(np.pi)
        self.training_units = frozenset()

    def fit(self, x, outcomes, unit_ids, iterations=40):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(outcomes, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimensions or y.shape != (len(x),) or len(x) == 0:
            raise ValueError("Nonempty, aligned feature matrix and binary outcomes required")
        if len(unit_ids) != len(x) or any(not u for u in unit_ids):
            raise ValueError("An episode ID is required for each training observation")
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Features must be finite and outcomes binary")
        precision = np.eye(self.dimensions) * self.prior_precision
        w = np.zeros(self.dimensions)
        def loss(candidate):
            z = x @ candidate
            return np.sum(np.logaddexp(0, z) - y * z) + self.prior_precision * (candidate @ candidate) / 2
        for _ in range(iterations):
            p = sigmoid(x @ w)
            gradient = x.T @ (p-y) + precision @ w
            hessian = x.T @ ((p*(1-p))[:, None] * x) + precision
            delta = np.linalg.solve(hessian, gradient)
            scale = 1.0
            before = loss(w)
            while scale > 1e-8 and loss(w-scale*delta) > before:
                scale *= .5
            w -= scale*delta
            if np.linalg.norm(scale*delta) < 1e-8:
                break
        p = sigmoid(x @ w)
        hessian = x.T @ ((p*(1-p))[:, None] * x) + precision
        self.mean = w
        self.covariance = np.linalg.solve(hessian, np.eye(self.dimensions))
        self.training_units = frozenset(unit_ids)
        return self

    def predict(self, x, *, unit_id, z=1.645):
        if not unit_id or unit_id in self.training_units:
            raise ValueError("Evaluation episode must be held out from fitting")
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimensions,) or not np.isfinite(x).all() or not np.isfinite(z) or z < 0:
            raise ValueError("Valid feature vector and nonnegative finite z required")
        location = float(x @ self.mean)
        variance = max(0.0, float(x @ self.covariance @ x))
        probabilities = sigmoid(location + np.sqrt(2*variance)*self.nodes)
        mean = float(self.weights @ probabilities)
        epistemic = float(self.weights @ ((probabilities-mean)**2))
        aleatoric = float(self.weights @ (probabilities*(1-probabilities)))
        return OutcomeBelief(float(sigmoid(location)), mean, epistemic, aleatoric,
                             float(sigmoid(location-z*np.sqrt(variance))),
                             float(sigmoid(location+z*np.sqrt(variance))))

    def sample_probability(self, x, rng):
        """Posterior sampling primitive for a future online exploration experiment."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimensions,) or not np.isfinite(x).all():
            raise ValueError("Valid feature vector required")
        return float(sigmoid(x @ rng.multivariate_normal(self.mean, self.covariance)))
