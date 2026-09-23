"""Normalized-unit residual adapter for a prospective protected-readout study.

This is output-unit engineering, not a new learning method. The immutable
qualified ProtectedReadout uses a raw-score residual; this thin subclass only
replaces its action_residual with a Linear subclass returning 64 * (W h + b).
The inherited centering therefore applies the centered residual in raw-score
units, mathematically 64 * ((W h + b) - mean(W h + b)). The implementation's
float32 order is explicitly scale first, then the parent's unchanged centering.
No alternate gradient scaling or optimizer is introduced. A later trainer owns
the common learning rate and loss normalization.

The six original backbone tensors contain5996 parameters. The residual retains
the same weight/bias names and116 parameters, for6112 total. Frozen mode trains
only116 parameters; joint mode trains6112. Both use the same scaled linear head,
zero initialization, inherited one-forward capture and no-feedback boundary.
The prediction/prior/carry fields and input validation remain inherited. Bitwise
backbone preservation requires matching parameter requires_grad flags and
gradient context, as documented by the qualified parent.

make_head(mode, seed) preserves the caller's CPU RNG and exactly reproduces the
original seeded backbone. from_pretrained(mode, seed, backbone_state) admits
exactly the six original CPU float32 finite tensors, copies their values without
aliasing or modifying the supplied state, and leaves the new residual zero.
There is no broad strict=False load or acceptance of extra/missing keys. The seed
remains initialization provenance, not a claim about the provenance of supplied
weights. This module loads no files containing weights or empirical data.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from openjev.research import otto_protected_readout as parent

VERSION = "otto-protected-training-model-v1"
PARENT_SOURCE_SHA256 = "3cfa99320472abad632cde112664be80212a7e283c44f4275e64d6c25c9a59e3"
SCALE = 64.0
BACKBONE_SHAPES = {
    "recurrent.weight_ih_l0": (84, 40),
    "recurrent.weight_hh_l0": (84, 28),
    "recurrent.bias_ih_l0": (84,),
    "recurrent.bias_hh_l0": (84,),
    "output.weight": (4, 28),
    "output.bias": (4,),
}
MODES = parent.MODES
BACKBONE_PARAMETERS, RESIDUAL_PARAMETERS, PARAMETERS = 5996, 116, 6112
Forecast, Carry, detach_carry = parent.Forecast, parent.Carry, parent.detach_carry
require = parent.require


def verify_parent():
    path = Path(parent.__file__)
    require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_SOURCE_SHA256,
            "immutable qualified protected-readout source pin")


class ScaledResidual(torch.nn.Linear):
    """Keep ordinary weight/bias tensors; emit normalized coordinates times64."""

    def forward(self, hidden):
        return SCALE * super().forward(hidden)


class ProtectedTrainingModel(parent.ProtectedReadout):
    def __init__(self, mode, seed):
        verify_parent()
        super().__init__(mode, seed)
        self.action_residual = ScaledResidual(parent.WIDTH, parent.SCORE_DIM,
                                              device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.action_residual.weight.zero_()
            self.action_residual.bias.zero_()


def parameter_count(mode, *, trainable_only=False):
    return parent.parameter_count(mode, trainable_only=trainable_only)


def make_head(mode, seed):
    """Create a fresh scaled residual with the unchanged seeded backbone."""
    require(mode in MODES and type(seed) is int and 0 <= seed < 2**32,
            "declared mode and uint32 seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ProtectedTrainingModel(mode, seed)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "exact total parameter count")
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)
            == parameter_count(mode, trainable_only=True), "exact trainable parameter count")
    return model


def from_pretrained(mode, seed, backbone_state):
    """Strictly copy original backbone weights into a fresh zero-residual model.

    Tensor validation completes before constructing or modifying a model. Copies
    occur under no_grad, so no graph is attached to supplied parameter tensors.
    Gradient locks are the selected mode's locks, not the source tensors' flags.
    """
    require(isinstance(backbone_state, Mapping) and set(backbone_state) == set(BACKBONE_SHAPES),
            "exact original backbone state keys")
    for name, shape in BACKBONE_SHAPES.items():
        value = backbone_state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and value.layout == torch.strided
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                "finite CPU float32 original backbone tensor: " + name)
    require(sum(value.numel() for value in backbone_state.values()) == BACKBONE_PARAMETERS,
            "exact original backbone parameter count")
    model = make_head(mode, seed)
    parameters = dict(model.named_parameters())
    require(set(parameters) == set(BACKBONE_SHAPES) | {"action_residual.weight", "action_residual.bias"},
            "exact destination parameter membership")
    with torch.no_grad():
        for name, value in backbone_state.items():
            parameters[name].copy_(value)
    require(bool((model.action_residual.weight == 0).all()) and bool((model.action_residual.bias == 0).all()),
            "pretrained copy leaves new residual zero")
    return model
