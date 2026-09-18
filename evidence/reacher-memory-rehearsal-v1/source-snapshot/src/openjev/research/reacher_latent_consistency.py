"""Engineering preview of an auxiliary public-history prediction objective.

This is not a V-JEPA reproduction, an RL algorithm, or evidence of useful
control. Existing observation/reward losses must remain separate. Only the
deterministic GRUWorldModel interface (including its reward-residual subclass)
is supported; stochastic RSSM state semantics are deliberately unsupported.

At root t, the student assimilates only packets through t. It then advances
through issued commands a[t:t+h] without future observations. A small head
predicts the frozen EMA teacher's hidden state after the teacher assimilates
the public packet at t+h. Future measurements are targets, never student
rollout inputs. Both root and endpoint must be valid public measurements.
The default seven-step horizon can bridge six consecutive missing packets:
from the last measurement at t to the next measurement at t+7. A ten-packet
gap requires an eleven-step horizon, which is an explicit configuration.

An optional prefix-valid transition mask marks padding after episode end.
The last valid transition may terminate an episode and its endpoint remains
available; no rollout crosses that endpoint. Concatenated episodes are not
accepted. Missing angular placeholders and padding are discarded before
either network sees them, including NaN placeholders.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor, nn

from openjev.research.reacher_world_models import GRUWorldModel, State


@dataclass(frozen=True)
class LatentPairs:
    """Tensors ordered [episode, root, hidden], with a [episode, root] mask."""

    prediction: Tensor
    target: Tensor
    valid: Tensor


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def collapse_diagnostics(features: Tensor) -> dict[str, float | int]:
    """Descriptive covariance effective rank; no claim about control sufficiency.

    The effective rank is exp(entropy(normalized covariance eigenvalues)).
    Constant features or fewer than two samples have rank zero. Calculations
    are detached and use CPU float64, including an explicit device transfer
    when needed; zero variance is not hidden by an epsilon. Diagnostic time
    and accelerator synchronization belong in end-to-end training accounting.
    """
    if features.ndim != 2 or features.shape[1] < 1 or not features.is_floating_point():
        raise ValueError("features must be floating [samples, dimensions]")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    values = features.detach().to(device="cpu", dtype=torch.float64)
    samples, dimensions = values.shape
    if samples < 2:
        return {"samples": samples, "dimensions": dimensions, "min_std": 0.0,
                "mean_std": 0.0, "effective_rank": 0.0}
    centered = values - values[:1]
    centered = centered - centered.mean(0)
    std = centered.square().mean(0).sqrt()
    spectrum = torch.linalg.svdvals(centered).square()
    mass = spectrum.sum()
    rank = 0.0
    if mass > 0:
        probability = spectrum[spectrum > 0] / mass
        rank = float(torch.exp(-(probability * probability.log()).sum()))
    return {"samples": samples, "dimensions": dimensions, "min_std": float(std.min()),
            "mean_std": float(std.mean()), "effective_rank": rank}


def vicreg_terms(features: Tensor, *, target_std: float = 1.0,
                 epsilon: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Separate student-only VICReg-style variance/covariance penalties.

    This is not the full two-view VICReg objective. Neighboring rollout
    endpoints are correlated, so these are optimization diagnostics, not
    uncertainty estimates. Callers may also use this helper on separately
    sampled cross-episode features. For fewer than two samples both penalties are undefined
    statistically and return differentiable zero; the caller must report that
    count, rather than interpreting zero as successful regularization.
    """
    target_std = _nonnegative(target_std, "target_std")
    epsilon = _nonnegative(epsilon, "epsilon")
    if epsilon == 0:
        raise ValueError("epsilon must be positive")
    if features.ndim != 2 or features.shape[1] < 1 or not features.is_floating_point():
        raise ValueError("features must be floating [samples, dimensions]")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    count, dimensions = features.shape
    if count < 2:
        zero = features.sum() * 0
        return zero, zero
    centered = features - features.mean(0)
    covariance = centered.T @ centered / (count - 1)
    variance = (target_std - (covariance.diagonal() + epsilon).sqrt()).clamp_min(0).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off_diagonal.square().sum() / dimensions


class LatentConsistencyAuxiliary(nn.Module):
    """An explicit EMA teacher and trainable hidden-to-hidden prediction head.

    The student is supplied to each call and is NOT registered twice. The
    optimizer must include both student parameters and ``predictor`` parameters.
    Call ``update_teacher(student)`` explicitly after a successful optimizer
    step. Forward never changes teacher weights. EMA applies to parameters;
    every named buffer, floating or integer and including nonpersistent
    buffers, is copied exactly from the student after schema validation.

    Prediction loss averages nonempty horizons equally. Optional regularizers
    operate only on valid student predictions and are zero-weight by default.
    No optimizer, data collection, model selection or planner is provided.
    A future runner must bind horizons, EMA/penalty settings and Python model
    configuration in checkpoint metadata; these are not state_dict tensors.
    """

    def __init__(self, student: GRUWorldModel, *, horizons: tuple[int, ...] = (1, 3, 7),
                 momentum: float = 0.99, variance_weight: float = 0.0,
                 covariance_weight: float = 0.0) -> None:
        super().__init__()
        if not isinstance(student, GRUWorldModel):
            raise TypeError("Only deterministic GRUWorldModel and its subclasses are supported")
        if (not isinstance(horizons, (tuple, list)) or not horizons
                or any(type(h) is not int or h < 1 for h in horizons)
                or len(set(horizons)) != len(horizons)):
            raise ValueError("horizons must be distinct positive integers")
        self.horizons = tuple(sorted(horizons))
        self.momentum = _nonnegative(momentum, "momentum")
        if self.momentum > 1:
            raise ValueError("momentum must be at most one")
        self.variance_weight = _nonnegative(variance_weight, "variance_weight")
        self.covariance_weight = _nonnegative(covariance_weight, "covariance_weight")
        self.teacher = copy.deepcopy(student)
        self._freeze_teacher()
        parameter = next(student.parameters())
        self.predictor = nn.Linear(student.hidden_size, student.hidden_size,
                                   device=parameter.device, dtype=parameter.dtype)

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        for parameter in self.teacher.parameters():
            parameter.grad = None

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def _check_student(self, student: GRUWorldModel) -> None:
        if type(student) is not type(self.teacher):
            raise TypeError("Student class changed after teacher construction")
        if student.hidden_size != self.teacher.hidden_size or student.dt != self.teacher.dt:
            raise ValueError("Student hidden size or timestep changed")
        for method in ("named_parameters", "named_buffers"):
            current = dict(getattr(student, method)())
            teacher = dict(getattr(self.teacher, method)())
            if current.keys() != teacher.keys():
                raise ValueError("Student/teacher tensor membership differs")
            if any((current[k].shape, current[k].dtype, current[k].device)
                   != (teacher[k].shape, teacher[k].dtype, teacher[k].device) for k in current):
                raise ValueError("Student/teacher tensor shape, dtype or device differs")

    @torch.no_grad()
    def update_teacher(self, student: GRUWorldModel) -> None:
        """Validate the entire schema before changing any teacher tensor."""
        self._check_student(student)
        current = dict(student.named_parameters())
        for name, parameter in self.teacher.named_parameters():
            parameter.mul_(self.momentum).add_(current[name], alpha=1 - self.momentum)
        current_buffers = dict(student.named_buffers())
        for name, buffer in self.teacher.named_buffers():
            buffer.copy_(current_buffers[name])
        self._freeze_teacher()

    def _inputs(self, student: GRUWorldModel, packets: Tensor, commands: Tensor,
                transition_valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        self._check_student(student)
        if commands.ndim != 3 or commands.shape[-1] != 2:
            raise ValueError("commands must be [batch, time, 2]")
        batch, steps, _ = commands.shape
        if batch < 1 or steps < 1 or packets.shape != (batch, steps + 1, 8):
            raise ValueError("packets must be [batch, time + 1, 8]")
        parameter = next(student.parameters())
        if any(not x.is_floating_point() or x.dtype != parameter.dtype or x.device != parameter.device
               for x in (packets, commands)):
            raise ValueError("Public inputs must match student floating dtype and device")
        if transition_valid is None:
            transition_valid = torch.ones(batch, steps, dtype=torch.bool, device=packets.device)
        if (transition_valid.shape != (batch, steps) or transition_valid.dtype != torch.bool
                or transition_valid.device != packets.device):
            raise ValueError("transition_valid must be boolean [batch, time] on the input device")
        if torch.any(transition_valid[:, 1:] & ~transition_valid[:, :-1]):
            raise ValueError("Each episode must have a prefix-valid transition mask; no restart/wrap")
        points = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=packets.device),
                            transition_valid), dim=1)
        public = torch.where(points[..., None], packets, 0)
        if not torch.all((public[..., 6] == 0) | (public[..., 6] == 1)):
            raise ValueError("Public measurement validity must be binary")
        angles = torch.where(public[..., 6:7] > 0.5, public[..., :4], 0)
        public = torch.cat((angles, public[..., 4:]), dim=-1)
        actions = torch.where(transition_valid[..., None], commands, 0)
        if not torch.isfinite(public).all() or not torch.isfinite(actions).all():
            raise ValueError("Available public features and issued commands must be finite")
        if torch.any(public[..., 7] < 0):
            raise ValueError("Public measurement age must be nonnegative")
        return public, actions, points

    @staticmethod
    def _history(model: GRUWorldModel, packets: Tensor, commands: Tensor) -> list[State]:
        state = model.initial(packets.shape[0], packets.device)
        roots = []
        for time in range(packets.shape[1]):
            state = model.assimilate(state, packets[:, time])
            roots.append(state)
            if time < commands.shape[1]:
                state, _, _ = model.advance(state, commands[:, time])
        return roots

    def pairs(self, student: GRUWorldModel, packets: Tensor, commands: Tensor, *,
              transition_valid: Tensor | None = None) -> dict[int, LatentPairs]:
        """Expose aligned synthetic-testable pairs, retaining student gradients."""
        public, actions, points = self._inputs(student, packets, commands, transition_valid)
        self._freeze_teacher()
        student_roots = self._history(student, public, actions)
        with torch.no_grad():
            teacher_roots = self._history(self.teacher, public, actions)
        predictions: dict[int, list[Tensor]] = {h: [] for h in self.horizons}
        steps = actions.shape[1]
        for root in range(steps):
            state = student_roots[root]
            for offset in range(1, min(max(self.horizons), steps - root) + 1):
                state, _, _ = student.advance(state, actions[:, root + offset - 1])
                if offset in predictions:
                    predictions[offset].append(self.predictor(state["hidden"]))
        result = {}
        for horizon in self.horizons:
            width = max(0, steps + 1 - horizon)
            if width:
                predicted = torch.stack(predictions[horizon], dim=1)
                target = torch.stack([state["hidden"] for state in teacher_roots[horizon:]], dim=1)
                valid = ((public[:, :width, 6] > 0.5) & (public[:, horizon:, 6] > 0.5)
                         & points[:, :width] & points[:, horizon:])
            else:
                predicted = public.new_empty(public.shape[0], 0, student.hidden_size)
                target = predicted.detach().clone()
                valid = points[:, :0]
            result[horizon] = LatentPairs(predicted, target.detach(), valid)
        return result

    def forward(self, student: GRUWorldModel, packets: Tensor, commands: Tensor, *,
                transition_valid: Tensor | None = None) -> tuple[Tensor, dict]:
        pairs = self.pairs(student, packets, commands, transition_valid=transition_valid)
        # The empty-target case still permits backward without inventing a label.
        zero = self.predictor.weight.sum() * 0 + next(student.parameters()).sum() * 0
        losses, predictions, targets, counts = [], [], [], {}
        for horizon, pair in pairs.items():
            count = int(pair.valid.sum())
            counts[str(horizon)] = count
            if count:
                prediction, target = pair.prediction[pair.valid], pair.target[pair.valid]
                losses.append((prediction - target).square().mean())
                predictions.append(prediction)
                targets.append(target)
        prediction_loss = torch.stack(losses).mean() if losses else zero
        predicted = torch.cat(predictions) if predictions else zero.expand(0, student.hidden_size)
        target = torch.cat(targets) if targets else predicted.detach()
        variance, covariance = vicreg_terms(predicted)
        loss = prediction_loss + self.variance_weight * variance + self.covariance_weight * covariance
        steps = commands.shape[1]
        metrics = {
            "engineering_preview": True, "prediction_loss": float(prediction_loss.detach()),
            "variance_penalty": float(variance.detach()), "covariance_penalty": float(covariance.detach()),
            "valid_pairs_by_horizon": counts, "nonempty_horizons": len(losses),
            "regularization_defined": predicted.shape[0] >= 2,
            "student": collapse_diagnostics(predicted), "teacher": collapse_diagnostics(target),
            "batch_size": commands.shape[0],
            "batch_forward_calls": {
                "student_assimilate": steps + 1, "student_prefix_advance": steps,
                "student_open_loop_advance": sum(min(max(self.horizons), steps - t) for t in range(steps)),
                "student_predictor": sum(max(0, steps + 1 - h) for h in self.horizons),
                "teacher_assimilate": steps + 1, "teacher_advance": steps,
            },
        }
        return loss, metrics
