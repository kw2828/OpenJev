"""Engineering comparator for action-conditioned public endpoint prediction.

The endpoint windows, masks and reduction match LatentConsistencyAuxiliary.
A hidden-to-hidden linear predictor has the same extra trainable parameter
count as that auxiliary; the student's existing observation_head then decodes
raw cosine/sine targets. This does not match gradient paths or total compute:
the shared observation decoder receives additional gradients, raw and latent
targets have different dimensions/scales, and this component has no teacher.
Equal scalar loss weights therefore do not imply equal gradient magnitudes.

Keep the existing native observation/reward objective separate and unchanged.
No optimizer, training runner, experiment protocol or performance claim is
provided. Only deterministic GRUWorldModel and its subclasses are supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from openjev.research.reacher_world_models import GRUWorldModel


@dataclass(frozen=True)
class RawEndpointPairs:
    """Tensors [episode, root, 4], with boolean [episode, root] validity."""

    prediction: Tensor
    target: Tensor
    valid: Tensor


def _model_configuration(student: GRUWorldModel) -> dict:
    configuration = {
        "class": f"{type(student).__module__}.{type(student).__qualname__}",
        "hidden_size": student.hidden_size,
        "dt": student.dt,
    }
    # These Python settings change residual-GRU behavior but are absent from
    # tensor-only state dictionaries. Do not silently accept a changed model.
    for name in ("residual_reward", "noise_std"):
        if hasattr(student, name):
            configuration[name] = getattr(student, name)
    return configuration


def _schema(student: GRUWorldModel) -> dict:
    return {
        method: tuple((name, tuple(value.shape)) for name, value in getattr(student, method)())
        for method in ("named_parameters", "named_buffers")
    }


class RawEndpointAuxiliary(nn.Module):
    """A separate prediction head; the supplied student is never registered.

    Optimizers must include the student and this auxiliary's predictor. The
    decoder remains owned by the student and must not be added a second time.
    Save configuration() alongside state_dict(): horizons and Python model
    settings are not tensor state. Moving both modules to a new dtype/device
    together is supported; their floating tensors must agree.

    Roots assimilate only their public prefix. An imagined window uses issued
    commands and no future packets. Only observed roots and observed endpoints
    of prefix-valid episodes contribute loss. Padding and missing angular
    placeholders are discarded before use, including NaNs. A terminal endpoint
    is allowed; no valid window crosses it. Padding still incurs computation,
    as in the latent component, and is included in the call counts.

    batch_forward_calls counts direct module/interface calls, including the
    additional endpoint decoder. Each standard GRU advance also executes its
    native observation and reward heads; those internal operations are paid
    within advances, not included again as extra endpoint decoder calls.
    sample_forward_evaluations multiplies these counts by the actual batch
    width. These counts are not FLOPs, backward costs, or wall-clock timings.
    """

    def __init__(self, student: GRUWorldModel, *, horizons: tuple[int, ...] = (1, 3, 7)) -> None:
        super().__init__()
        if not isinstance(student, GRUWorldModel):
            raise TypeError("Only deterministic GRUWorldModel and its subclasses are supported")
        if (not isinstance(horizons, (tuple, list)) or not horizons
                or any(type(h) is not int or h < 1 for h in horizons)
                or len(set(horizons)) != len(horizons)):
            raise ValueError("horizons must be distinct positive integers")
        self.horizons = tuple(sorted(horizons))
        self._student_type = type(student)
        self._student_configuration = _model_configuration(student)
        self._student_schema = _schema(student)
        parameter = next(student.parameters())
        self.predictor = nn.Linear(student.hidden_size, student.hidden_size,
                                   device=parameter.device, dtype=parameter.dtype)

    def configuration(self) -> dict:
        """Fresh JSON-compatible metadata required alongside predictor tensors."""
        return {
            "component": "raw_endpoint_auxiliary_v1",
            "engineering_preview": True,
            "horizons": list(self.horizons),
            "model": dict(self._student_configuration),
            "predictor": "linear_hidden_to_hidden",
            "extra_trainable_parameters": sum(p.numel() for p in self.predictor.parameters()),
            "decoder": "student.observation_head",
            "decoder_receives_auxiliary_gradients": True,
            "targets": ["cos(q0)", "cos(q1)", "sin(q0)", "sin(q1)"],
            "target_dimensions": 4,
            "target_detached": True,
            "mask": "observed_root_and_endpoint_with_prefix_valid_transitions",
            "reduction": "mean_of_nonempty_horizon_per_dimension_mse",
            "teacher": None,
            "regularization": None,
            "dtype": str(self.predictor.weight.dtype),
            "device": str(self.predictor.weight.device),
            "parameter_count_matched_not_compute_or_gradient_path": True,
        }

    def _check_student(self, student: GRUWorldModel) -> None:
        if type(student) is not self._student_type:
            raise TypeError("Student class changed after auxiliary construction")
        if _model_configuration(student) != self._student_configuration:
            raise ValueError("Student Python configuration changed")
        if _schema(student) != self._student_schema:
            raise ValueError("Student tensor membership or shape changed")
        for parameter in student.parameters():
            if (parameter.dtype, parameter.device) != (self.predictor.weight.dtype,
                                                       self.predictor.weight.device):
                raise ValueError("Student and predictor dtype/device must agree")

    def _inputs(self, student: GRUWorldModel, packets: Tensor, commands: Tensor,
                transition_valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        self._check_student(student)
        if commands.ndim != 3 or commands.shape[-1] != 2:
            raise ValueError("commands must be [batch, time, 2]")
        batch, steps, _ = commands.shape
        if batch < 1 or steps < 1 or packets.shape != (batch, steps + 1, 8):
            raise ValueError("packets must be [batch, time + 1, 8]")
        parameter = self.predictor.weight
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

    def _pairs_and_counts(self, student: GRUWorldModel, packets: Tensor, commands: Tensor,
                          transition_valid: Tensor | None) -> tuple[dict[int, RawEndpointPairs], dict]:
        public, actions, points = self._inputs(student, packets, commands, transition_valid)
        calls = dict.fromkeys(("student_assimilate", "student_prefix_advance",
                              "student_open_loop_advance", "student_predictor",
                              "student_endpoint_decoder"), 0)
        state = student.initial(public.shape[0], public.device)
        roots = []
        steps = actions.shape[1]
        for time in range(steps + 1):
            state = student.assimilate(state, public[:, time])
            calls["student_assimilate"] += 1
            roots.append(state)
            if time < steps:
                state, _, _ = student.advance(state, actions[:, time])
                calls["student_prefix_advance"] += 1
        predictions: dict[int, list[Tensor]] = {h: [] for h in self.horizons}
        for root in range(steps):
            state = roots[root]
            for offset in range(1, min(max(self.horizons), steps - root) + 1):
                state, _, _ = student.advance(state, actions[:, root + offset - 1])
                calls["student_open_loop_advance"] += 1
                if offset in predictions:
                    projected = self.predictor(state["hidden"])
                    calls["student_predictor"] += 1
                    predictions[offset].append(student.observation_head(projected))
                    calls["student_endpoint_decoder"] += 1
        result = {}
        for horizon in self.horizons:
            width = max(0, steps + 1 - horizon)
            if width:
                predicted = torch.stack(predictions[horizon], dim=1)
                target = public[:, horizon:, :4].detach()
                valid = ((public[:, :width, 6] > 0.5) & (public[:, horizon:, 6] > 0.5)
                         & points[:, :width] & points[:, horizon:])
            else:
                predicted = public.new_empty(public.shape[0], 0, 4)
                target = predicted.detach().clone()
                valid = points[:, :0]
            result[horizon] = RawEndpointPairs(predicted, target, valid)
        return result, calls

    def pairs(self, student: GRUWorldModel, packets: Tensor, commands: Tensor, *,
              transition_valid: Tensor | None = None) -> dict[int, RawEndpointPairs]:
        """Aligned raw endpoint pairs; gradients flow through predictions only."""
        return self._pairs_and_counts(student, packets, commands, transition_valid)[0]

    def forward(self, student: GRUWorldModel, packets: Tensor, commands: Tensor, *,
                transition_valid: Tensor | None = None) -> tuple[Tensor, dict]:
        pairs, calls = self._pairs_and_counts(student, packets, commands, transition_valid)
        zero = self.predictor.weight.sum() * 0 + next(student.parameters()).sum() * 0
        losses, counts = [], {}
        for horizon, pair in pairs.items():
            count = int(pair.valid.sum())
            counts[str(horizon)] = count
            if count:
                prediction, target = pair.prediction[pair.valid], pair.target[pair.valid]
                if not torch.isfinite(prediction).all():
                    raise ValueError("Valid endpoint predictions must be finite")
                losses.append((prediction - target).square().mean())
        loss = torch.stack(losses).mean() if losses else zero
        return loss, {
            "engineering_preview": True,
            "prediction_loss": float(loss.detach()),
            "valid_pairs_by_horizon": counts,
            "nonempty_horizons": len(losses),
            "batch_size": commands.shape[0],
            "batch_forward_calls": calls,
            "sample_forward_evaluations": {name: value * commands.shape[0] for name, value in calls.items()},
            "configuration": self.configuration(),
        }
