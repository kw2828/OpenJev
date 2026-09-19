"""One retrospective conditional probe over frozen, independently rolled experts.

The probe observes poses0..26, fits support rows1..25, then conditions on the
five recorded action blocks26..30. Poses27..31 are used only after forecasting
to score it. Those action blocks need not have been knowable at origin26, so
this is not a claim of an online forecast issued at that historical instant.
All state belongs to this call. Proper rotations and authenticated frozen
weights are caller responsibilities; no rotation projection is performed.
"""
from __future__ import annotations

from time import perf_counter

import torch
from torch import Tensor

from openjev.research.pose_coordination import blend
from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import PoseTransport
from openjev.research.rigid_motion import geodesic_angle, so3_log

VARIANTS = ("fast", "slow", "half", "probe_half", "probe_inverse", "probe_fit")


def _check(fast, slow, p, rotation, past):
    if type(fast) is not PoseSupport or type(slow) is not PoseTransport:
        raise ValueError("exact PoseSupport and PoseTransport classes required")
    if fast.mode != "learned" or slow.variant != "body" or slow.frame != "body" or slow.transport:
        raise ValueError("learned support and untransported body GRU required")
    if (fast.scales.dtype not in (torch.float32, torch.float64)
            or slow.scales.dtype != fast.scales.dtype or slow.scales.device != fast.scales.device):
        raise ValueError("experts require matching float32/64 dtype and device")
    if not isinstance(p, Tensor) or p.ndim != 3 or p.shape[0] < 1:
        raise ValueError("context positions require a positive batch")
    batch = p.shape[0]
    fast._check(p, (batch, 32, 3), "context positions")
    fast._check(rotation, (batch, 32, 3, 3), "context rotations")
    fast._check(past, (batch, 31, 40), "past actions")


def _prefix_fit(fast: PoseSupport, p: Tensor, rotation: Tensor, actions: Tensor):
    """Exactly27 poses/26 actions; support row s targets pose s+1, s=1..25.

    This small extension uses the frozen feature and weighted-normal helpers.
    Full32 fitting below always delegates the original, unmodified fit path.
    """
    batch = p.shape[0]
    fast._check(p, (batch, 27, 3), "probe positions")
    fast._check(rotation, (batch, 27, 3, 3), "probe rotations")
    fast._check(actions, (batch, 26, 40), "probe past actions")
    dp = p[:, 1:] - p[:, :-1]
    w = so3_log(rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2))
    current = rotation[:, 1:26]
    inverse = current.transpose(-1, -2)
    x = fast._features(current, dp[:, :25], w[:, :25], actions[:, 1:26]).double()
    y = torch.cat(((inverse @ (dp[:, 1:] - dp[:, :-1]).unsqueeze(-1)).squeeze(-1) / fast.scales[2],
                   (inverse @ (w[:, 1:] - w[:, :-1]).unsqueeze(-1)).squeeze(-1) / fast.scales[3]), -1).double()
    prior = fast.prior.double()
    posterior = prior.unsqueeze(0).expand(batch, -1, -1)
    base = torch.exp2(-torch.arange(24, -1, -1, dtype=torch.float64, device=p.device) / 5)
    for _ in range(3):
        residual = (y - x @ posterior).abs()
        weights = base[None, :, None] * (1.5 / residual.clamp_min(1.5))
        posterior = fast._solve(x, y, prior, weights)
    return {"p": p[:, 26].clone(), "R": rotation[:, 26].clone(),
            "dp": dp[:, -1].clone(), "w": w[:, -1].clone(),
            "weights": posterior.to(dtype=fast.scales.dtype).clone()}


def _slow_rollout(slow, state, actions):
    # Frozen advance returns fresh state; cloning also isolates the caller's root.
    state = {key: value.clone() for key, value in state.items()}
    positions, rotations = [], []
    for action in actions.unbind(1):
        state = slow.advance(state, action)
        positions.append(state["p"])
        rotations.append(state["R"])
    result = torch.stack(positions, 1), torch.stack(rotations, 1)
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise ValueError("nonfinite slow forecast")
    return result


def _slow_context(slow, p, rotation, past, *, probe):
    state = slow.assimilate(slow.initial(p[:, 0], rotation[:, 0]), p[:, 0], rotation[:, 0])
    prediction = None
    for t in range(31):
        state = slow.advance(state, past[:, t])
        state = slow.assimilate(state, p[:, t + 1], rotation[:, t + 1])
        if probe and t == 25:
            prediction = _slow_rollout(slow, state, past[:, 26:31])
    # Unlike PoseTransport.forecast(), do not replace actual root motion by CV16.
    return state, prediction


def _fallback(value: Tensor, reference: Tensor) -> Tensor:
    batch = reference.shape[0]
    if not (isinstance(value, Tensor) and value.shape in ((2,), (batch, 2))
            and value.dtype == reference.dtype and value.device == reference.device
            and bool(torch.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all())):
        raise ValueError("fallback_alpha requires finite [2] or [B,2] fractions in the input dtype/device")
    return value.expand(batch, 2).clone()


def probe_context(fast: PoseSupport, slow: PoseTransport, p32: Tensor,
                  r32: Tensor, past31: Tensor) -> dict:
    """Return slow_state, fast_p/fast_R/slow_p/slow_R and timing diagnostics.

    No target comparison occurs here. Fast fitting receives only27 poses and
    26 actions. The GRU probe branches after assimilation26; its private five
    advances never assimilate poses27..31. The real context then continues to
    root31 unchanged. Returned tensors belong to this call, without an external
    cache. They are mutable tensors; gradients are not detached.
    """
    start = perf_counter()
    _check(fast, slow, p32, r32, past31)
    tick = perf_counter()
    state = _prefix_fit(fast, p32[:, :27], r32[:, :27], past31[:, :26])
    fast_p, fast_r = fast.rollout(state, past31[:, 26:31])
    fast_seconds = perf_counter() - tick
    tick = perf_counter()
    slow_state, (slow_p, slow_r) = _slow_context(slow, p32, r32, past31, probe=True)
    return {"slow_state": slow_state, "fast_p": fast_p, "fast_R": fast_r,
            "slow_p": slow_p, "slow_R": slow_r,
            "diagnostics": {"fast_fit_and_probe_seconds": fast_seconds,
                            "slow_context_and_probe_seconds": perf_counter() - tick,
                            "wall_seconds": perf_counter() - start}}


def _inverse_alpha(errors: Tensor, fallback: Tensor):
    """Opposite error ratio after common rescaling, including subnormal errors."""
    scale = errors.amax(1)
    zero = scale == 0
    scaled = errors / torch.where(zero, torch.ones_like(scale), scale)[:, None]
    total = scaled.sum(1)
    alpha = torch.where(zero, fallback.double(), scaled[:, 1] / torch.where(zero, torch.ones_like(total), total))
    return alpha.to(dtype=fallback.dtype), zero


def choose_alpha(fast_p: Tensor, fast_r: Tensor, slow_p: Tensor, slow_r: Tensor,
                 target_p: Tensor, target_r: Tensor, *, variant: str,
                 fallback_alpha: Tensor) -> tuple[Tensor, dict]:
    """Pure completed-probe scoring, returning fractions and tensor evidence.

    All errors use float64 metrics, averaging five squared Euclidean/geodesic
    distances, not per-coordinate MSE. Position LS uses mean dot products.
    Rotation fit evaluates all17 frozen production blends in the input dtype,
    then uses the float64 metric; the first exact minimum wins. No projection,
    confidence interpretation, optimizer, randomness or future targets exist.
    """
    from openjev.research.pose_coordination import _check as check_tensor

    if variant not in ("probe_half", "probe_inverse", "probe_fit"):
        raise ValueError("choose_alpha requires a probe variant")
    if not isinstance(fast_p, Tensor) or fast_p.ndim != 3 or len(fast_p) < 1:
        raise ValueError("positive probe batch required")
    batch = len(fast_p)
    for name, value, tail in (("fast_p", fast_p, (3,)), ("slow_p", slow_p, (3,)),
                              ("target_p", target_p, (3,)), ("fast_R", fast_r, (3, 3)),
                              ("slow_R", slow_r, (3, 3)), ("target_R", target_r, (3, 3))):
        check_tensor(value, (batch, 5, *tail), fast_p, name)
    fallback = _fallback(fallback_alpha, fast_p)
    errors = torch.stack([torch.stack(((p.double() - target_p.double()).square().sum(-1).mean(1),
                                       geodesic_angle(r.double(), target_r.double()).square().mean(1)), -1)
                          for p, r in ((fast_p, fast_r), (slow_p, slow_r))], 1)
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("nonfinite probe errors")
    evidence = {"errors": errors}
    if variant == "probe_inverse":
        alpha, zero = _inverse_alpha(errors, fallback)
        evidence["inverse_zero_fallback"] = zero
    else:
        delta = fast_p.double() - slow_p.double()
        residual = target_p.double() - slow_p.double()
        numerator = (delta * residual).sum(-1).mean(1)
        denominator = delta.square().sum(-1).mean(1)
        threshold = (torch.finfo(fast_p.dtype).eps ** 2 * errors[:, :, 0].amax(1)).clamp_min(
            torch.finfo(torch.float64).tiny)
        tiny = denominator <= threshold
        fraction = numerator / torch.where(tiny, torch.ones_like(denominator), denominator)
        position = torch.where(tiny, fallback[:, 0].double(), fraction.clamp(0, 1)).to(dtype=fast_p.dtype)
        grid = torch.arange(17, dtype=fast_p.dtype, device=fast_p.device) / 16
        grid_alpha = grid[None, :, None].expand(batch, 17, 2).reshape(batch * 17, 2)
        _, grid_r = blend(*(v.repeat_interleave(17, dim=0) for v in (fast_p, fast_r, slow_p, slow_r)), grid_alpha)
        grid_r = grid_r.reshape(batch, 17, 5, 3, 3)
        grid_mse = geodesic_angle(grid_r.double(), target_r[:, None].double().expand_as(grid_r)).square().mean(2)
        index = grid_mse.argmin(1)
        fitted_alpha = torch.stack((position, grid[index]), -1)
        alpha = torch.full_like(fitted_alpha, .5) if variant == "probe_half" else fitted_alpha
        evidence.update(position_numerator=numerator, position_denominator=denominator,
                        position_threshold=threshold, position_fallback=tiny,
                        grid_coefficients=grid, grid_rotations=grid_r,
                        grid_mse=grid_mse, grid_index=index, fitted_alpha=fitted_alpha)
    if not bool(torch.isfinite(alpha).all()) or not all(bool(torch.isfinite(x).all()) for x in evidence.values()):
        raise ValueError("nonfinite coefficient selection")
    evidence["alpha"] = alpha
    return alpha, evidence


def forecast(fast: PoseSupport, slow: PoseTransport, variant: str, p32: Tensor,
             r32: Tensor, past31: Tensor, future25: Tensor, *, fallback_alpha: Tensor,
             return_probe: bool = False, on_probe=None):
    """Return (p,R,alpha,JSON diagnostics), plus probe arrays if requested.

    return_probe=True always returns five items, with None for fast/slow/half.
    on_probe, if supplied, synchronously receives owned copies of the four
    probe prediction arrays BEFORE target scoring, so the caller can save and
    seal them. Its work/copy time is charged. It receives no root/target/model;
    callback mutation cannot alter predictions. Callback errors propagate.

    probe_half computes and retains the same fitted coefficient as probe_fit,
    then discards it for(.5,.5). Inverse deliberately skips the grid work.
    Full32 fast fitting and full-batch25 expert rollouts use their unchanged
    frozen paths. All mixed variants compute both experts even at endpoint
    fractions. CPU wall timings include validation/copy/scoring/JSON preparation
    but exclude caller saving after return. Counts describe completed successful
    calls, not partial failure work; this helper has no checkpoint or I/O system.
    """
    start = perf_counter()
    if variant not in VARIANTS or type(return_probe) is not bool or (on_probe is not None and not callable(on_probe)):
        raise ValueError("invalid variant, probe flag or callback")
    _check(fast, slow, p32, r32, past31)
    batch = len(p32)
    fast._check(future25, (batch, 25, 40), "future actions")
    fallback = _fallback(fallback_alpha, p32)
    probed = variant.startswith("probe_")
    probe_arrays = None
    probe_times = None
    selection = None
    context_seconds = callback_seconds = scoring_seconds = 0.0
    if probed:
        result = probe_context(fast, slow, p32, r32, past31)
        slow_state, probe_times = result["slow_state"], result["diagnostics"]
        probe_arrays = {key: result[key] for key in ("fast_p", "fast_R", "slow_p", "slow_R")}
        if on_probe is not None:
            tick = perf_counter()
            on_probe({key: value.detach().clone() for key, value in probe_arrays.items()})
            callback_seconds = perf_counter() - tick
        tick = perf_counter()
        alpha, evidence = choose_alpha(*probe_arrays.values(), p32[:, 27:32], r32[:, 27:32],
                                       variant=variant, fallback_alpha=fallback)
        # Grid rotations remain numeric artifact payload, not large JSON lists.
        selection = {key: value.detach().cpu().tolist() for key, value in evidence.items() if key != "grid_rotations"}
        probe_arrays.update(evidence)
        scoring_seconds = perf_counter() - tick
    else:
        alpha = p32.new_full((batch, 2), 1. if variant == "fast" else 0. if variant == "slow" else .5)
        if variant != "fast":
            tick = perf_counter()
            slow_state, _ = _slow_context(slow, p32, r32, past31, probe=False)
            context_seconds = perf_counter() - tick
    tick = perf_counter()
    if variant != "slow":
        state = fast.fit_context(p32, r32, past31, "decay_huber3")
        fp, fr = fast.rollout(state, future25)
    if variant != "fast":
        sp, sr = _slow_rollout(slow, slow_state, future25)
    position, rotation = ((fp, fr) if variant == "fast" else (sp, sr) if variant == "slow"
                          else blend(fp, fr, sp, sr, alpha))
    forecast_seconds = perf_counter() - tick
    nfast, nslow = batch * (variant != "slow"), batch * (variant != "fast")
    diagnostics = {"version": "pose-probe-blend-v1", "variant": variant,
                   "probe": {"origin": 26, "support_indices": [1, 25], "support_rows": 25,
                             "target_indices": [27, 31], "action_indices": [26, 30],
                             "age_half_life": 5, "huber_delta": 1.5, "huber_iterations": 3,
                             "ridge_precision": 1.0, "solve_dtype": "float64"} if probed else None,
                   "fallback_alpha": fallback.detach().cpu().tolist(),
                   "alpha": alpha.detach().cpu().tolist(), "selection": selection,
                   "timing": {"probe": probe_times, "slow_context_seconds": context_seconds,
                              "callback_seconds": callback_seconds, "scoring_seconds": scoring_seconds,
                              "full_forecast_seconds": forecast_seconds},
                   "work": {"scope": "successful calls; no timing or partial-failure guarantee",
                            "fast_support_rows": 25 * batch * probed + 30 * nfast,
                            "fast_feature_samples": 30 * batch * probed + 55 * nfast,
                            "fast_batched_solve_calls": 3 * probed + 3 * bool(nfast),
                            "fast_cholesky_factorizations": 18 * (batch * probed + nfast),
                            "fast_forecast_samples": 5 * batch * probed + 25 * nfast,
                            "slow_observation_samples": 32 * nslow,
                            "slow_context_advance_samples": 31 * nslow,
                            "slow_probe_advance_samples": 5 * batch * probed,
                            "slow_forecast_advance_samples": 25 * nslow,
                            "grid_rotation_samples": 85 * batch * (variant in ("probe_fit", "probe_half")),
                            "final_blend_samples": 25 * batch * (variant not in ("fast", "slow")),
                            "probe_callback_calls": int(probed and on_probe is not None)}}
    diagnostics["wall_seconds"] = perf_counter() - start
    result = (position, rotation, alpha, diagnostics)
    return (*result, probe_arrays) if return_probe else result
