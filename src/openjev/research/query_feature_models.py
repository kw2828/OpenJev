"""Public-context Bayesian linear mixtures over learned nonlinear features.

Each block defines a separate posterior under w ~ N(0, I). The uniform-prior
block identity is inferred from the JOINT few-shot predictive density. Each
component then conditions on those few shots once. Calls are stateless: replay
means recomputing this same conditional distribution, not adding likelihood.

Centered models encode all inputs relative to the current query. They define
query-conditional predictors, not a claim of one consistent joint process over
different query-dependent feature maps. Gaussian uncertainty is model-based;
this module makes no empirical calibration or architectural novelty claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

VERSION = "query-feature-models-v1"
NOISE_VARIANCE = .0225
NYSTROM_JITTER = 1e-6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, name, ndim):
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.float64 and value.ndim == ndim,
             f"{name} must be a CPU float64 tensor with {ndim} dimensions")
    _require(bool(torch.isfinite(value).all()), f"finite {name}")


def _noise(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             "noise_variance must be finite and positive")
    return float(value)


class GaussianMixturePrediction(dict):
    """Four tensor fields and a stable mixture log density for held-out targets."""

    def log_prob(self, target):
        _tensor(target, "target", 1)
        _require(target.shape == self["component_mean"].shape[:1], "target batch shape")
        variance = self["component_variance"]
        error = target[:, None] - self["component_mean"]
        component = -.5 * (math.log(2 * math.pi) + variance.log() + error.square() / variance)
        value = torch.logsumexp(self["log_weights"] + component, dim=1)
        _require(bool(torch.isfinite(value).all()), "finite mixture log_prob")
        return value


@dataclass(frozen=True)
class FeatureArchive:
    """Owned static-feature tensors, valid only for an unchanged model.

    tensor_bytes includes features and all four posterior tensors, without
    double-counting model parameters. Python metadata and autograd's temporary
    training graph are not included, so this is not peak process memory.
    """

    block_features: torch.Tensor
    precision: torch.Tensor
    eta: torch.Tensor
    cholesky: torch.Tensor
    weight_mean: torch.Tensor
    owner: int
    signature: tuple

    @property
    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.block_features, self.precision, self.eta, self.cholesky, self.weight_mean))


def _block_posterior(block_features, block_targets, variance):
    rank = block_features.shape[-1]
    eye = torch.eye(rank, dtype=torch.float64, device="cpu")
    transpose = block_features.transpose(-1, -2)
    precision = eye + (transpose @ block_features) / variance
    eta = (transpose @ block_targets[..., None]) / variance
    _require(bool(torch.isfinite(precision).all() and torch.isfinite(eta).all()),
             "finite block sufficient statistics")
    chol = torch.linalg.cholesky(precision)
    weight_mean = torch.cholesky_solve(eta, chol)
    return precision, eta, chol, weight_mean


def gaussian_feature_mixture(block_features, block_targets, fewshot_features,
                             fewshot_targets, query_features, *, noise_variance=NOISE_VARIANCE):
    """Infer a mixture from feature tensors, with shapes B,K,N,R; B,K,N;
    B,F,R; B,F; B,R. N and F count distinct observations, even if values match.

    No inverse, variance clamp, diagonal likelihood approximation or posterior
    cache is used. Positive noise and the identity prior make the exact systems
    positive definite even for rank-deficient feature matrices. Numerical
    non-finiteness or a failed Cholesky factorization is an error, not a retry.
    """
    variance = _noise(noise_variance)
    for value, name, ndim in ((block_features, "block_features", 4),
                              (block_targets, "block_targets", 3),
                              (fewshot_features, "fewshot_features", 3),
                              (fewshot_targets, "fewshot_targets", 2),
                              (query_features, "query_features", 2)):
        _tensor(value, name, ndim)
    b, k, n, rank = block_features.shape
    f = fewshot_features.shape[1]
    _require(min(b, k, n, rank, f) > 0, "positive batch, block, context, rank and few-shot sizes")
    _require(block_targets.shape == (b, k, n)
             and fewshot_features.shape == (b, f, rank)
             and fewshot_targets.shape == (b, f) and query_features.shape == (b, rank),
             "matching feature/target shapes")
    posterior = _block_posterior(block_features, block_targets, variance)
    return _predict_features(*posterior, fewshot_features, fewshot_targets, query_features, variance)


def _predict_features(precision, eta, chol, weight_mean, fewshot_features,
                      fewshot_targets, query_features, variance):
    b, k, rank, _ = precision.shape
    f = fewshot_features.shape[1]

    # Predict all F labels jointly under each block's posterior, before any of
    # these labels is assimilated. Shared weight uncertainty correlates them.
    few = fewshot_features[:, None].expand(b, k, f, rank)
    whitened = torch.linalg.solve_triangular(chol, few.transpose(-1, -2), upper=False)
    covariance = variance * torch.eye(f, dtype=torch.float64, device="cpu")
    covariance = covariance + whitened.transpose(-1, -2) @ whitened
    residual = fewshot_targets[:, None, :, None] - few @ weight_mean
    _require(bool(torch.isfinite(covariance).all() and torch.isfinite(residual).all()),
             "finite joint few-shot predictive law")
    predictive_chol = torch.linalg.cholesky(covariance)
    standardized = torch.linalg.solve_triangular(predictive_chol, residual, upper=False)
    log_evidence = -.5 * (f * math.log(2 * math.pi)
                          + 2 * predictive_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                          + standardized.square().sum((-2, -1)))
    _require(bool(torch.isfinite(log_evidence).all()), "finite block log evidence")
    log_weights = torch.log_softmax(log_evidence, dim=1)

    # Add the few-shot sufficient statistics exactly once to each component.
    few_transpose = fewshot_features.transpose(-1, -2)
    updated_precision = precision + (few_transpose @ fewshot_features)[:, None] / variance
    updated_eta = eta + (few_transpose @ fewshot_targets[..., None])[:, None] / variance
    _require(bool(torch.isfinite(updated_precision).all() and torch.isfinite(updated_eta).all()),
             "finite conditioned sufficient statistics")
    updated_chol = torch.linalg.cholesky(updated_precision)
    updated_mean = torch.cholesky_solve(updated_eta, updated_chol)
    query = query_features[:, None, :, None].expand(b, k, rank, 1)
    component_mean = (query.transpose(-1, -2) @ updated_mean).squeeze(-1).squeeze(-1)
    query_whitened = torch.linalg.solve_triangular(updated_chol, query, upper=False)
    component_variance = variance + query_whitened.square().sum((-2, -1))
    _require(bool(torch.isfinite(component_mean).all() and torch.isfinite(component_variance).all()
                  and (component_variance > 0).all() and torch.isfinite(log_weights).all()),
             "finite means, log weights and positive component variances")
    prob_positive = (log_weights.exp()
                     * torch.special.ndtr(component_mean / component_variance.sqrt())).sum(1)
    _require(bool(torch.isfinite(prob_positive).all()
                  and (prob_positive >= 0).all() and (prob_positive <= 1 + 1e-12).all()),
             "finite mixture sign probability")
    return GaussianMixturePrediction(component_mean=component_mean,
                                     component_variance=component_variance,
                                     log_weights=log_weights, prob_positive=prob_positive)


class _FeatureMixture(nn.Module):
    def __init__(self, rank, centered, noise_variance):
        super().__init__()
        _require(type(rank) is int and rank > 0, "positive integer rank")
        _require(type(centered) is bool, "Boolean centered mode")
        self.rank, self.centered = rank, centered
        self.noise_variance = _noise(noise_variance)

    def forward(self, bx, by, fx, fy, qx):
        for value, name, ndim in ((bx, "bx", 4), (by, "by", 3), (fx, "fx", 3),
                                  (fy, "fy", 2), (qx, "qx", 2)):
            _tensor(value, name, ndim)
        b, k, n, dim = bx.shape
        f = fx.shape[1]
        _require(min(b, k, n, f) > 0 and dim == 2 and by.shape == (b, k, n)
                 and fx.shape == (b, f, 2) and fy.shape == (b, f) and qx.shape == (b, 2),
                 "public context/query shapes")
        if self.centered:
            bx = bx - qx[:, None, None, :]
            fx = fx - qx[:, None, :]
            qx = qx - qx
        # Encode together so Nyström builds its inducing factor exactly once
        # per forward call and all three feature groups share that factor.
        inputs = torch.cat((bx.reshape(b, k * n, 2), fx, qx[:, None]), dim=1)
        phi = self.features(inputs)
        return gaussian_feature_mixture(phi[:, :k * n].reshape(b, k, n, self.rank), by,
                                         phi[:, k * n:k * n + f], fy, phi[:, -1],
                                         noise_variance=self.noise_variance)

    def _signature(self):
        return (self.rank, self.centered, self.noise_variance,
                tuple((id(x), x._version) for x in (*self.parameters(), *self.buffers())))

    def encode_archive(self, bx, by):
        """Fit each public block once for a static, unchanged encoder.

        Re-encode after any parameter/buffer update. Caches preserve autograd;
        use no_grad for inference storage measurements. Never mutate tensors
        inside the returned archive. Centered models cannot share this cache.
        """
        _require(not self.centered, "query-centered models cannot cache archive features")
        _tensor(bx, "bx", 4)
        _tensor(by, "by", 3)
        b, k, n, dim = bx.shape
        _require(min(b, k, n) > 0 and dim == 2 and by.shape == (b, k, n), "archive shapes")
        features = self.features(bx)
        posterior = _block_posterior(features, by, self.noise_variance)
        return FeatureArchive(features, *posterior, owner=id(self), signature=self._signature())

    def predict_cached(self, cache, fx, fy, qx):
        """Condition once on this request's few shots without changing cache."""
        _require(not self.centered, "query-centered models cannot cache archive features")
        _require(isinstance(cache, FeatureArchive) and cache.owner == id(self)
                 and cache.signature == self._signature(), "archive owner or stale model parameters")
        for value, name, ndim in ((fx, "fx", 3), (fy, "fy", 2), (qx, "qx", 2)):
            _tensor(value, name, ndim)
        b = cache.block_features.shape[0]
        f = fx.shape[1]
        _require(f > 0 and fx.shape == (b, f, 2) and fy.shape == (b, f) and qx.shape == (b, 2),
                 "cached request shapes")
        phi = self.features(torch.cat((fx, qx[:, None]), dim=1))
        return _predict_features(cache.precision, cache.eta, cache.cholesky, cache.weight_mean,
                                  phi[:, :f], fy, phi[:, -1], self.noise_variance)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class QueryFeatureModel(_FeatureMixture):
    """Learned 2 -> 32 tanh -> rank features divided by sqrt(rank)."""

    def __init__(self, rank=16, centered=False, seed=0, noise_variance=NOISE_VARIANCE):
        super().__init__(rank, centered, noise_variance)
        _require(type(seed) is int and 0 <= seed < 2**32, "uint32 local seed")
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(2, 32, device="cpu", dtype=torch.float64),
                                         nn.Tanh(),
                                         nn.Linear(32, rank, device="cpu", dtype=torch.float64))

    def features(self, inputs):
        _require(isinstance(inputs, torch.Tensor) and inputs.ndim >= 2, "feature input dimensions")
        _tensor(inputs, "feature inputs", inputs.ndim)
        _require(inputs.shape[-1] == 2, "two-dimensional feature inputs")
        result = self.encoder(inputs) / math.sqrt(self.rank)
        _require(bool(torch.isfinite(result).all()), "finite learned features")
        return result


class NystromFeatureModel(_FeatureMixture):
    """Trainable RBF length/amplitude with a fixed 4x4 inducing grid.

    K(x,z)=amplitude^2 exp(-||x-z||^2/(2 length^2)). Features are
    K(x,Z) chol(K(Z,Z)+1e-6 I)^(-T), so their inner products form the
    positive semidefinite Nyström approximation. No diagonal correction is
    added. Centering changes coordinates relative to the same fixed grid.
    """

    def __init__(self, rank=16, centered=False, noise_variance=NOISE_VARIANCE):
        _require(type(rank) is int and rank == 16, "Nyström rank must be16 for the fixed4x4 grid")
        super().__init__(rank, centered, noise_variance)
        coordinates = torch.linspace(-2., 2., 4, dtype=torch.float64, device="cpu")
        self.register_buffer("inducing_grid", torch.cartesian_prod(coordinates, coordinates))
        self.log_length = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))
        self.log_amplitude = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))

    def features(self, inputs):
        _require(isinstance(inputs, torch.Tensor) and inputs.ndim >= 2, "feature input dimensions")
        _tensor(inputs, "feature inputs", inputs.ndim)
        _require(inputs.shape[-1] == 2, "two-dimensional feature inputs")
        length = self.log_length.exp()
        amplitude_squared = (2 * self.log_amplitude).exp()
        _require(bool(torch.isfinite(length) and (length > 0)
                      and torch.isfinite(amplitude_squared) and (amplitude_squared > 0)),
                 "finite positive RBF length and amplitude squared")

        def kernel(left, right):
            distance = ((left[..., :, None, :] - right) / length).square().sum(-1)
            result = amplitude_squared * torch.exp(-.5 * distance)
            _require(bool(torch.isfinite(result).all()), "finite RBF kernel")
            return result

        grid = self.inducing_grid
        kzz = kernel(grid, grid) + NYSTROM_JITTER * torch.eye(16, dtype=torch.float64, device="cpu")
        chol = torch.linalg.cholesky(kzz)
        flat = inputs.reshape(-1, 2)
        features = torch.linalg.solve_triangular(chol, kernel(flat, grid).T, upper=False).T
        result = features.reshape(*inputs.shape[:-1], self.rank)
        _require(bool(torch.isfinite(result).all()), "finite Nyström features")
        return result
