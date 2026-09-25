"""One supervised sensitivity-learning update on caller-supplied tensors.

No dataset access, scheduling, checkpoint selection or study runner. This uses
the unchanged residual architecture. A failed optimizer update is not repaired;
the caller must retain it as a failed attempt rather than retry this model.
"""
from __future__ import annotations

import math

import torch

from openjev.research import fsm_sensitivity as sensitivity
from openjev.research.fsm_residual import FSMResidual, _require

TANGENT_ARMS = ('total_sensitivity', 'max_envelope', 'relative_sensitivity')


def _model(model):
    _require(type(model) is FSMResidual and model.mode == 'feedback'
             and model.residual_kind == 'tanh', 'tanh feedback model required')
    model._validate_parameters()
    _require(all(p.requires_grad for p in model.parameters()), 'all residual parameters trainable')


def make_optimizer(model, *, learning_rate=3e-4):
    """Create the ordinary Adam comparator shared by every arm."""
    _model(model)
    _require(type(learning_rate) in (int, float) and math.isfinite(learning_rate)
             and learning_rate > 0, 'positive finite learning rate')
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.)


def _optimizer(model, optimizer):
    _require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
             'single-group ordinary Adam required')
    group = optimizer.param_groups[0]
    parameters = tuple(model.parameters())
    _require(len(group['params']) == len(parameters)
             and all(a is b for a, b in zip(group['params'], parameters, strict=True)),
             'optimizer must own exactly this model in parameter order')
    _require(group['weight_decay'] == 0 and tuple(group['betas']) == (.9, .999)
             and group['eps'] == 1e-8 and not group['amsgrad'] and not group['maximize']
             and not group['differentiable'] and not group['capturable']
             and group['foreach'] is None and group['fused'] is None, 'ordinary Adam options required')
    _require(type(group['lr']) in (int, float) and math.isfinite(group['lr']) and group['lr'] > 0,
             'positive finite Adam learning rate')
    _require(all(any(key is p for p in parameters) for key in optimizer.state), 'foreign optimizer state')
    for slots in optimizer.state.values():
        for value in slots.values():
            if isinstance(value, torch.Tensor):
                _require(bool(torch.isfinite(value).all()), 'nonfinite Adam state; no repair')


def train_step(model, optimizer, y_context, u_context, future_u, target, *,
               arm, penalty_weight, directions=None, horizons=sensitivity.HORIZONS,
               gradient_clip=1.):
    """Apply one complete update and return scalars, with no retained graph.

    All arms condition on the same observed lags. Targets enter only the MSE;
    future observations never enter initialization, the forecast or its tangents.
    The caller owns paired batches/directions and any study-level time budget.
    Simple arms require no directions and perform no tangent work. The three
    sensitivity arms use the same full learned and linear tangent propagation.
    """
    _model(model)
    _optimizer(model, optimizer)
    _require(type(arm) is str and arm in sensitivity.ARMS, 'declared arm required')
    _require(type(penalty_weight) in (int, float) and math.isfinite(penalty_weight)
             and penalty_weight >= 0 and (arm != 'unregularized' or penalty_weight == 0),
             'finite nonnegative penalty weight; zero for unregularized')
    _require(type(gradient_clip) in (int, float) and math.isfinite(gradient_clip)
             and gradient_clip > 0, 'positive finite gradient clip')
    data = (y_context, u_context, future_u, target)
    _require(all(isinstance(t, torch.Tensor) and not t.requires_grad for t in data),
             'caller data must be tensors without autograd state')
    _require(future_u.ndim == 3 and future_u.shape[1] > 0, 'positive forecast horizon')
    model._tensor(target, tuple(future_u.shape), 'supervised target')
    state = model.condition(y_context, u_context)
    use_tangents = arm in TANGENT_ARMS
    if use_tangents:
        _require(isinstance(directions, torch.Tensor) and not directions.requires_grad,
                 'explicit detached paired directions required')
    else:
        _require(directions is None, 'simple arm does not consume tangent directions')
    result = (sensitivity.rollout_tangents(model, future_u, state, directions) if use_tangents
              else sensitivity.rollout_primal(model, future_u, state))
    mse = (result['prediction'] - target).square().mean()
    penalty = sensitivity.regularizer(model, result, arm, horizons)
    loss = mse + penalty_weight * penalty
    _require(bool(torch.isfinite(loss)), 'nonfinite learning objective; no update')
    hinge_active_fraction = None
    if arm in ('max_envelope', 'relative_sensitivity'):
        gain, baseline = sensitivity.gain_profiles(result, horizons)
        allowance = baseline if arm == 'relative_sensitivity' else baseline.max(dim=1, keepdim=True).values
        hinge_active_fraction = float((gain > allowance).to(torch.float64).mean())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = tuple(model.parameters())
    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
             'nonfinite or missing residual gradient; no update')
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    _require(all(bool(torch.isfinite(p.grad).all()) for p in parameters), 'nonfinite clipped gradient; no update')
    optimizer.step()
    model._validate_parameters()
    _optimizer(model, optimizer)
    batch, horizon = len(state), future_u.shape[1]
    tangent_steps = batch * directions.shape[1] * horizon if use_tangents else 0
    return {'arm': arm, 'mse': float(mse.detach()), 'penalty': float(penalty.detach()),
            'penalty_weight': float(penalty_weight), 'objective': float(loss.detach()),
            'gradient_norm_before_clip': float(norm), 'gradient_clip': float(gradient_clip),
            'hinge_active_fraction': hinge_active_fraction, 'completed_updates': 1,
            'work': {'primal_window_steps': batch*horizon, 'learned_tangent_direction_steps': tangent_steps,
                     'linear_tangent_direction_steps': tangent_steps, 'backward_calls': 1, 'optimizer_steps': 1},
            'scope': 'One caller-supplied supervised update. No stability, generalization or novelty claim.'}
