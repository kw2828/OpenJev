"""Algebraic deployment of a frozen VARX with affine feedback correction.

This conversion performs no fitting. In real arithmetic, adding the two linear
maps preserves every free-running forecast. Floating-point reassociation can
change last bits, so callers must qualify forecast parity for their checkpoints.
The returned plain VARXModel retains no residual module or Torch tensors.
"""
from __future__ import annotations

import numpy as np

from .fsm_linear import VARXModel
from .fsm_residual import FSMResidual


def fold_affine_feedback(residual: FSMResidual, parent: VARXModel) -> VARXModel:
    """Return an independent merged VARX, preserving explicit parent provenance.

    The parent's order, ridge alpha, fit-row count and FIT identities describe
    the original backbone fit, not a new ridge fit to the merged coefficients.
    No head, gradients, source array, module, prediction or lag state is retained.
    Only affine feedback is supported: output-only residuals follow a different
    latent linear trajectory and cannot use this same coefficient addition.
    """
    if type(residual) is not FSMResidual or type(parent) is not VARXModel:
        raise ValueError('exact FSMResidual and explicit VARXModel parent required')
    residual._validate_parameters()
    if residual.residual_kind != 'affine' or residual.mode != 'feedback':
        raise ValueError('only affine feedback can be folded into this VARX recurrence')
    # Revalidate all provenance and finite float64 geometry, even if a caller
    # bypassed the frozen dataclass contract after the original construction.
    checked = VARXModel(parent.coefficients, parent.order, parent.alpha,
                        parent.fit_rows, parent.fit_record_ids)
    if residual.order != checked.order or not np.array_equal(
            residual.coefficients.detach().numpy(), checked.coefficients):
        raise ValueError('residual backbone must exactly match the explicit parent')
    merged = checked.coefficients.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        merged[:, :-1] += residual.residual.weight.detach().numpy()
        merged[:, -1] += residual.residual.bias.detach().numpy()
    if not np.isfinite(merged).all():
        raise ValueError('nonfinite folded coefficients; no clipping or repair')
    return VARXModel(merged, checked.order, checked.alpha,
                     checked.fit_rows, checked.fit_record_ids)
