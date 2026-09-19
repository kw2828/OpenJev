"""Fixed support weighting for the frozen learned PoseAdaptation checkpoint.

No new parameters, persistent cache, training, or uncertainty interpretation.
All weighted solves are float64, including intermediate IRLS iterates; only
the returned posterior is cast to model dtype. Mass controls pay for the full
base fit before discarding its posterior and fitting uniformly weighted data.
"""
from __future__ import annotations

import torch
from torch import Tensor

from openjev.research.pose_adaptation import PoseAdaptation

VARIANTS = (
    "prior", "full", "recent5", "recent5_mass", "decay5", "decay5_mass",
    "huber3", "huber3_mass", "decay_huber3", "decay_huber3_mass",
)


class PoseSupport(PoseAdaptation):
    """Load the original learned meta state_dict strictly, without conversion."""

    def __init__(self, scales: Tensor):
        super().__init__("learned", scales)

    @staticmethod
    def _solve(phi: Tensor, targets: Tensor, prior: Tensor, weights: Tensor) -> Tensor:
        """One batched Cholesky call, with six independent normals per case."""
        normal = torch.einsum("bnf,bno,bng->bofg", phi, weights, phi)
        normal = normal + torch.eye(phi.shape[-1], dtype=phi.dtype, device=phi.device)
        residual = targets - phi @ prior
        rhs = torch.einsum("bnf,bno,bno->bof", phi, weights, residual)
        correction = torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(normal))
        result = prior + correction.squeeze(-1).transpose(-1, -2)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite weighted posterior")
        return result

    def _weighted_fit(self, phi: Tensor, targets: Tensor, variant: str):
        """Private raw-weight witness for synthetic checks, not JSON artifacts.

        Returns posterior, actual weights, derivation weights, discarded/base
        posterior and solve-call count. The final IRLS weights are those used
        in solve three, computed from iterate two, not a fourth reweighting.
        """
        x, y, prior = phi.double(), targets.double(), self.prior.double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("nonfinite support features or targets")
        base = variant.removesuffix("_mass")
        ages = torch.arange(29, -1, -1, dtype=torch.float64, device=x.device)
        if base == "recent5":
            age_weights = (ages < 5).double()
        elif base in ("decay5", "decay_huber3"):
            age_weights = torch.exp2(-ages / 5)
        elif base == "huber3":
            age_weights = torch.ones_like(ages)
        else:
            raise ValueError("unknown weighted support variant")
        age_weights = age_weights[None, :, None].expand(x.shape[0], -1, 6)
        posterior = prior.unsqueeze(0).expand(x.shape[0], -1, -1)
        calls = 3 if "huber" in base else 1
        for _ in range(calls):
            weights = age_weights
            if "huber" in base:
                residual = (y - x @ posterior).abs()
                # Equivalent to min(1, 1.5 / |residual|), finite at zero.
                weights = age_weights * (1.5 / residual.clamp_min(1.5))
            posterior = self._solve(x, y, prior, weights)
        derivation_weights, derivation_posterior = weights, posterior
        if variant.endswith("_mass"):
            weights = (weights.sum(1, keepdim=True) / 30).expand(-1, 30, -1).clone()
            posterior = self._solve(x, y, prior, weights)
            calls += 1
        return posterior, weights, derivation_weights, derivation_posterior, calls

    @staticmethod
    def _weight_summary(weights: Tensor, *, no_fit: bool = False):
        sums = weights.sum(1)
        squared = weights.square().sum(1)
        effective = torch.where(squared > 0, sums.square() / squared.clamp_min(1e-300),
                                torch.zeros_like(sums))
        below = torch.zeros_like(sums) if no_fit else (weights < 1).double().mean(1)
        return {"weight_sum": sums.detach().cpu().tolist(),
                "effective_n": effective.detach().cpu().tolist(),
                "fraction_weight_lt_one": below.detach().cpu().tolist()}

    def fit_context(self, p: Tensor, rotation: Tensor, actions: Tensor,
                    variant: str = "full", *, return_diagnostics: bool = False):
        """Exactly 32 poses/31 actions; no future-pose or persistent-state input.

        Support t=1..30 has age 29..0. The root is actual pose31 and motion30->31.
        Prior/full delegate the original fit path exactly. Diagnostic counts
        describe successful calls; they are not timers or failed-prefix counts.
        Prior summaries are zero, including fraction below one by the explicit
        no-fit convention. All other summaries describe actual support weights.
        """
        if variant not in VARIANTS or type(return_diagnostics) is not bool:
            raise ValueError("invalid support variant or diagnostics flag")
        if not isinstance(p, Tensor) or p.ndim != 3 or p.shape[0] < 1:
            raise ValueError("context positions require positive batch size")
        batch = p.shape[0]
        self._check(p, (batch, 32, 3), "context positions")
        self._check(rotation, (batch, 32, 3, 3), "context rotations")
        self._check(actions, (batch, 31, 40), "context actions")
        if variant in ("prior", "full"):
            state = super().fit_context(p, rotation, actions, adapt=variant == "full")
            if not return_diagnostics:
                return state
            actual = torch.full((batch, 30, 6), float(variant == "full"),
                                dtype=torch.float64, device=p.device)
            derivation = actual
            derivation_posterior = state["weights"]
            calls = int(variant == "full")
            factorizations = batch * calls
        else:
            phi, targets, dp, w = self._context_design(p, rotation, actions)
            posterior, actual, derivation, derivation_posterior, calls = self._weighted_fit(phi, targets, variant)
            state = {"p": p[:, -1].clone(), "R": rotation[:, -1].clone(),
                     "dp": dp.clone(), "w": w.clone(),
                     "weights": posterior.to(dtype=self.scales.dtype).clone()}
            if not all(bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError("nonfinite fitted pose state")
            if not return_diagnostics:
                return state
            factorizations = 6 * batch * calls
        with torch.no_grad():
            prior = self.prior.double()
            correction = torch.linalg.vector_norm(state["weights"].double() - prior, dim=(1, 2))
            # Describe the discarded base arm in the same returned dtype as
            # that arm itself, not its pre-cast Cholesky intermediate.
            derived = torch.linalg.vector_norm(
                derivation_posterior.to(dtype=self.scales.dtype).double() - prior, dim=(1, 2))
            diagnostics = {
                "version": "pose-support-v1", "variant": variant,
                "base_variant": variant.removesuffix("_mass"), "adapted": variant != "prior",
                "mass_control": variant.endswith("_mass"), "support_rows": 30,
                "huber_iterations": 3 if "huber" in variant else 0,
                "ridge_precision": 1.0, "solve_dtype": "float64",
                "batched_solve_calls": calls, "cholesky_factorizations": factorizations,
                "actual": self._weight_summary(actual, no_fit=variant == "prior"),
                "derivation": self._weight_summary(derivation, no_fit=variant == "prior"),
                "correction_frobenius_norm": correction.cpu().tolist(),
                "derivation_correction_frobenius_norm": derived.cpu().tolist(),
            }
        return state, diagnostics

    def configuration(self):
        return {**super().configuration(), "support_adapter": "pose-support-v1",
                "variants": list(VARIANTS), "recent_supports": [26, 27, 28, 29, 30],
                "age_half_life": 5, "huber_delta": 1.5, "huber_iterations": 3,
                "huber_initialization": "original global prior",
                "mass_control": "derive full base fit, then uniform final support mass per output",
                "prior_weight_summary": "zero mass, effective_n and fraction below one; no fit",
                "intermediate_solve_dtype": "float64; cast final posterior to model dtype",
                "work_scope": "successful solve calls/factorizations only; no measured speed claim"}
