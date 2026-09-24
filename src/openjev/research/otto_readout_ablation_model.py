"""Three parameter masks over the unchanged period-four numerical predictor.

Engineering only: no dataset, file loading, optimization or study admission.
All arms own the same eight tensors and6112 values. They train116,232 or6112
parameters respectively. Frozen GRU parameters do NOT disable autograd through
the recurrent operations: trainable base readouts can affect later innovations.

The scheduled kernel and no-memory composition forward are inherited unchanged.
Only constructors, parameter membership and the scheduled validation entrypoint
differ. Full joint matches the qualified joint model. Cross-arm bitwise equality
is not promised because gradient flags can alter native GRU rounding. The
residual-only state invariant compares its own flags before/after readout edits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_model as composed
from openjev.research import otto_scheduled_predictor as predictor

VERSION = "otto-readout-ablation-model-v1"
ARMS = ("action_residual_only", "both_readouts", "full_joint")
TRAINABLE_COUNTS = dict(zip(ARMS, (116, 232, 6112), strict=True))
SOURCE_PINS = {composed: "0c7cafb8bb9401d876a39262c63379a02356611f24a02768eb9fdf8dda42a126",
               predictor: "e30c3e311bdde24e1dd5692e29e74578412744728afb3f2d9fb93de09eeaaf27"}
Carry, Forecast, detach_carry = composed.Carry, composed.Forecast, composed.detach_carry
require = memory.require


def verify_sources():
    for module, expected in SOURCE_PINS.items():
        require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected,
                "unchanged qualified source: " + module.__name__)
    composed.verify_sources()


def configuration(arm, seed, query_period):
    require(type(arm) is str and arm in ARMS and type(seed) is int and 0 <= seed < 2**32,
            "declared ablation arm and uint32 seed")
    require(type(query_period) is int and query_period == 4, "readout ablation is period four only")


def trainable_names(arm):
    require(type(arm) is str and arm in ARMS, "declared parameter mask")
    return tuple(name for name in predictor.STATE_SHAPES
                 if arm == "full_joint" or name.startswith("action_residual.")
                 or (arm == "both_readouts" and name.startswith("output.")))


class ReadoutPredictor(predictor.ScheduledPredictor):
    """Keep gradient context active for all three masks, including feedback."""

    def __init__(self, arm, seed, query_period=4):
        configuration(arm, seed, query_period)
        verify_sources()
        super().__init__("joint", seed, query_period)
        self._arm = arm
        allowed = set(trainable_names(arm))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in allowed)
        self.validate_parameters()

    @property
    def arm(self):
        return self._arm

    def validate_parameters(self):
        configuration(self.arm, self.seed, self.query_period)
        named = dict(self.named_parameters())
        allowed = set(trainable_names(self.arm))
        require(self.mode == "joint" and self.kind == predictor.KIND
                and set(named) == set(predictor.STATE_SHAPES)
                and all(p.requires_grad == (name in allowed) and p.device.type == "cpu"
                        and p.dtype == torch.float32 and tuple(p.shape) == predictor.STATE_SHAPES[name]
                        for name, p in named.items()), "unchanged ablation shapes and gradient locks")
        require(sum(p.numel() for p in named.values() if p.requires_grad) == TRAINABLE_COUNTS[self.arm],
                "exact ablation trainable count")

    def forward(self, features, query_scores, lengths, query_mask, *, carry=None, episode_ends):
        require(self._action_lock.acquire(blocking=False), "ablation predictor cannot be reentered or used concurrently")
        try:
            self.validate_parameters()
            require(self._captured is self._action_hidden is None, "no stale inherited capture")
            # No no_grad boundary here: readout gradients must cross fixed GRU
            # operations when the base readout changes query innovation.
            return self._scheduled_forward(features, query_scores, lengths, query_mask,
                                           carry=carry, episode_ends=episode_ends)
        finally:
            self._action_lock.release()


class ReadoutAblationModel(composed.QueryMemoryModel):
    """No-memory composition with inherited forward/carry and exact ownership."""

    def __init__(self, arm, seed=0, query_period=4):
        configuration(arm, seed, query_period)
        verify_sources()
        torch.nn.Module.__init__(self)
        self.slow = ReadoutPredictor(arm, seed, query_period)
        self.config, self.seed = memory.Config("none", key_dim=8), seed
        self.memory, self.projection = memory.QueryMemory(self.config), None

    @property
    def arm(self):
        return self.slow.arm

    @property
    def query_period(self):
        return self.slow.query_period

    def effective_named_parameters(self):
        require(type(self.slow) is ReadoutPredictor and self.config == memory.Config("none", key_dim=8)
                and self.projection is None and self.seed == self.slow.seed,
                "unchanged no-memory ablation composition")
        self.slow.validate_parameters()
        allowed = set(trainable_names(self.arm))
        return [("slow." + name, value) for name, value in self.slow.named_parameters() if name in allowed]


def make_model(arm, seed=0, query_period=4):
    return ReadoutAblationModel(arm, seed, query_period)


def from_state(arm, seed, slow_state, query_period=4):
    """Copy exactly eight in-memory CPU float32 tensors without input aliases."""
    configuration(arm, seed, query_period)
    composed.validate_slow_state(slow_state)
    model = make_model(arm, seed, query_period)
    model.slow.load_state_dict(slow_state, strict=True)
    return model
