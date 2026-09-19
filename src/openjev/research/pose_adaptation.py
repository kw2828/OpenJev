"""Causal fixed-ridge pose adaptation with learned or public linear features.

Rotations map body to world; dp and w are world displacement (meters/step)
and spatial rotation increment (radians/step). The six regression targets are
changes of those increments, expressed in the current body frame. They are
not body-twist differences across differently oriented frames. Only completed
context transitions fit weights, which remain fixed throughout imagination.

This is an ALPaCA-style deterministic ridge model, not calibrated Bayesian
uncertainty or a new regression principle. Proper input rotations are assumed.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from openjev.research.rigid_motion import so3_exp, so3_log


def _rotate(rotation: Tensor, vector: Tensor) -> Tensor:
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


class PoseAdaptation(nn.Module):
    """Context32 -> physical root and fixed linear weights; action-only rollout.

    scales contains four positive scalars: displacement, angular increment,
    displacement change and angular-increment change. The caller derives and
    authenticates them using training data only. Actions are already normalized.
    """

    def __init__(self, mode: str, scales: Tensor):
        super().__init__()
        if mode not in ("learned", "public"):
            raise ValueError("mode must be learned or public")
        if not (isinstance(scales, Tensor) and scales.shape == (4,)
                and scales.dtype in (torch.float32, torch.float64)
                and bool(torch.isfinite(scales).all()) and bool((scales > 0).all())):
            raise ValueError("scales must contain four finite positive float32/64 scalars")
        self.mode = mode
        self.feature_dim = 13 if mode == "learned" else 50
        self.register_buffer("scales", scales.detach().clone())
        if mode == "learned":
            self.encoder = nn.Sequential(nn.Linear(49, 48), nn.Tanh(),
                                         nn.Linear(48, 12), nn.Tanh())
            self.encoder.to(dtype=scales.dtype, device=scales.device)
        else:
            self.encoder = None
        self.prior = nn.Parameter(torch.zeros(self.feature_dim, 6, dtype=scales.dtype,
                                             device=scales.device))

    def _check(self, value: Tensor, shape: tuple[int, ...], name: str):
        if not (isinstance(value, Tensor) and tuple(value.shape) == shape
                and value.dtype == self.scales.dtype and value.device == self.scales.device
                and bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} requires finite shape {shape} and model dtype/device")

    def _features(self, rotation: Tensor, dp: Tensor, w: Tensor, action: Tensor) -> Tensor:
        inverse = rotation.transpose(-1, -2)
        inputs = torch.cat((rotation[..., 2, :], _rotate(inverse, dp) / self.scales[0],
                            _rotate(inverse, w) / self.scales[1], action), dim=-1)
        encoded = self.encoder(inputs) if self.encoder is not None else inputs.tanh()
        normalized = F.normalize(encoded, p=2, dim=-1, eps=1e-8)
        return torch.cat((normalized, torch.ones_like(normalized[..., :1])), dim=-1)

    def _context_design(self, p: Tensor, rotation: Tensor, actions: Tensor):
        dp = p[:, 1:] - p[:, :-1]
        w = so3_log(rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2))
        current_rotation = rotation[:, 1:31]
        phi = self._features(current_rotation, dp[:, :30], w[:, :30], actions[:, 1:31])
        inverse = current_rotation.transpose(-1, -2)
        targets = torch.cat((_rotate(inverse, dp[:, 1:] - dp[:, :-1]) / self.scales[2],
                             _rotate(inverse, w[:, 1:] - w[:, :-1]) / self.scales[3]), -1)
        return phi, targets, dp[:, -1], w[:, -1]

    def fit_context(self, p: Tensor, rotation: Tensor, actions: Tensor, *,
                    adapt: bool = True, return_diagnostics: bool = False):
        """Fit exactly30 supports t=1..30 from32 poses and31 completed actions.

        Support t uses action[t], backward motion (t-1)->t, and the completed
        successor t+1. Action0 has no support because no earlier motion exists.
        The root is pose31. No future observations are accepted. adapt=False
        uses only the learned global prior and performs no regression solve.

        Diagnostics are detached JSON-compatible scalars/lists; requested
        condition numbers use one eigenspectrum but no additional solve.
        """
        if type(adapt) is not bool or type(return_diagnostics) is not bool:
            raise ValueError("adapt and return_diagnostics must be booleans")
        if not isinstance(p, Tensor) or p.ndim != 3 or p.shape[0] < 1:
            raise ValueError("context positions must have positive batch size")
        batch = p.shape[0]
        self._check(p, (batch, 32, 3), "context positions")
        self._check(rotation, (batch, 32, 3, 3), "context rotations")
        self._check(actions, (batch, 31, 40), "context actions")
        normal = None
        phi = targets = None
        if adapt:
            phi, targets, dp, w = self._context_design(p, rotation, actions)
            x, y, prior = phi.double(), targets.double(), self.prior.double()
            transpose = x.transpose(-1, -2)
            normal = transpose @ x + torch.eye(self.feature_dim, dtype=torch.float64,
                                               device=x.device)
            rhs = transpose @ (y - x @ prior)
            correction = torch.cholesky_solve(rhs, torch.linalg.cholesky(normal))
            weights = (prior + correction).to(dtype=self.scales.dtype)
        else:
            dp = p[:, -1] - p[:, -2]
            w = so3_log(rotation[:, -1] @ rotation[:, -2].transpose(-1, -2))
            weights = self.prior.unsqueeze(0).expand(batch, -1, -1).clone()
        state = {"p": p[:, -1].clone(), "R": rotation[:, -1].clone(),
                 "dp": dp.clone(), "w": w.clone(), "weights": weights.clone()}
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("nonfinite fitted pose state")
        if not return_diagnostics:
            return state
        with torch.no_grad():
            condition = None
            residual = None
            if normal is not None:
                spectrum = torch.linalg.eigvalsh(normal.detach())
                condition = (spectrum[:, -1] / spectrum[:, 0]).cpu().tolist()
                residual = (targets - phi @ weights).square().mean((1, 2)).cpu().tolist()
            diagnostics = {"adapt": adapt, "available_support_transitions": 30,
                           "fitted_support_transitions": 30 if adapt else 0,
                           "ridge_precision": 1.0, "solve_dtype": "float64",
                           "cholesky_factorizations": batch if adapt else 0,
                           "batched_solve_calls": 1 if adapt else 0,
                           "normal_condition_number": condition,
                           "support_mean_squared_residual": residual,
                           "correction_frobenius_norm": torch.linalg.vector_norm(
                               weights - self.prior, dim=(1, 2)).cpu().tolist()}
        return state, diagnostics

    def rollout(self, state: dict[str, Tensor], future_actions: Tensor) -> tuple[Tensor, Tensor]:
        """Return poses [B,H,3]/[B,H,3,3], using the same weights at every step.

        Increment changes are predicted in the pre-step body frame, rotated
        into world coordinates, then integrated: dp+=R*ddp, w+=R*dw,
        p+=dp, R=Exp(w)*R. No pseudo-observation updates are performed.
        """
        if not isinstance(state, dict) or set(state) != {"p", "R", "dp", "w", "weights"}:
            raise ValueError("pose state keys do not match")
        if (not isinstance(future_actions, Tensor) or future_actions.ndim != 3
                or future_actions.shape[0] < 1 or future_actions.shape[1] < 1):
            raise ValueError("future actions require positive batch and horizon")
        batch, horizon = future_actions.shape[:2]
        self._check(future_actions, (batch, horizon, 40), "future actions")
        for key, tail in (("p", (3,)), ("R", (3, 3)), ("dp", (3,)),
                          ("w", (3,)), ("weights", (self.feature_dim, 6))):
            self._check(state[key], (batch, *tail), key)
        p, rotation, dp, w = (state[key].clone() for key in ("p", "R", "dp", "w"))
        weights = state["weights"]
        positions, rotations = [], []
        for action in future_actions.unbind(1):
            phi = self._features(rotation, dp, w, action)
            change = (phi.unsqueeze(-2) @ weights).squeeze(-2)
            dp = dp + _rotate(rotation, change[:, :3] * self.scales[2])
            w = w + _rotate(rotation, change[:, 3:] * self.scales[3])
            p = p + dp
            rotation = so3_exp(w) @ rotation
            positions.append(p)
            rotations.append(rotation)
        result = torch.stack(positions, 1), torch.stack(rotations, 1)
        if not all(bool(torch.isfinite(value).all()) for value in result):
            raise ValueError("nonfinite pose rollout")
        return result

    def configuration(self):
        return {"class": type(self).__name__, "mode": self.mode, "context_poses": 32,
                "support_transitions": 30, "input_dim": 49, "feature_dim": self.feature_dim,
                "encoder": [49, 48, 12] if self.mode == "learned" else "tanh_public49",
                "feature_normalization": "L2 eps1e-8 then append intercept",
                "ridge_precision": 1.0, "solve_dtype": "float64",
                "scales": self.scales.detach().cpu().tolist(),
                "targets": "current-body changes of world displacement and spatial rotation",
                "forecast_weight_updates": 0, "uncertainty_claim": False}

    def parameter_and_state_counts(self):
        scalars = 18 + self.feature_dim * 6
        return {"trainable_parameters": sum(p.numel() for p in self.parameters()),
                "persistent_state_scalars_per_case": scalars,
                "persistent_state_bytes_per_case": scalars * self.scales.element_size(),
                "scope": "tensor payload only; excludes support, solve, autograd and rollout temporaries"}
