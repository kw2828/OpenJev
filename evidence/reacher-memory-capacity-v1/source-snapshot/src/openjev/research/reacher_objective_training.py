"""Deterministic training primitives for the prospective objective ablation.

No corpus, fitting loop, evaluation, planner, or scientific result lives here.
CPU float32, complete episodes and the existing anchor objective are deliberate
constraints. Partial attempts are terminal; checkpoint restoration is provided
for fresh post-training evaluation and synthetic state-parity tests, not a
permission to resume a failed study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from openjev.research.reacher_latent_consistency import LatentConsistencyAuxiliary
from openjev.research.reacher_raw_endpoint import RawEndpointAuxiliary
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import sequence_loss

ARMS = ("anchor", "raw", "latent")
BACKBONE_PREFIXES = ("observation_update.", "transition.")
ADAM = {
    "name": "Adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
    "decoupled_weight_decay": False,
}
CALIBRATION = {
    "batches": 4,
    "target_fraction": 0.5,
    "clip": [0.01, 100.0],
    "parameters": ["observation_update", "transition"],
}
VERSION = "reacher-objective-training-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, *, minimum=0.0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, f"Invalid {name}")
    return float(value)


def clone(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return copy.deepcopy(value)


def canonical_tensor_hash(values: Mapping[str, Tensor]) -> str:
    """SHA256 header + sorted, length-prefixed JSON metadata and raw CPU bytes.

    Header is b'OpenJev named tensors v1\\0'. For each sorted UTF-8 name:
    metadata=json.dumps([name,str(dtype),list(shape)],ensure_ascii=False,
    separators=(',',':')).encode('utf-8'); append big-endian uint64 length,
    metadata, big-endian uint64 byte count, contiguous uint8 tensor bytes.
    This deliberately does not depend on torch.save ZIP metadata or key order.
    """
    require(
        isinstance(values, Mapping) and all(isinstance(key, str) for key in values),
        "Named tensor mapping required",
    )
    digest = hashlib.sha256(b"OpenJev named tensors v1\0")
    for name in sorted(values):
        value = values[name]
        require(isinstance(value, Tensor) and value.layout == torch.strided, "Dense tensor required")
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Nonfinite tensor")
        metadata = json.dumps(
            [name, str(value.dtype), list(value.shape)], ensure_ascii=False, separators=(",", ":")
        ).encode()
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(metadata)) + metadata + struct.pack(">Q", len(raw)) + raw)
    return digest.hexdigest()


def canonical_state_hash(value) -> str:
    """Bind nested optimizer/configuration structure as well as all tensor bytes."""

    def description(item):
        if isinstance(item, Tensor):
            return ["tensor", canonical_tensor_hash({"value": item})]
        if isinstance(item, dict):
            require(all(type(key) in (str, int) for key in item), "Unsupported checkpoint key")
            return [
                "dict",
                [
                    [type(key).__name__, key, description(item[key])]
                    for key in sorted(item, key=lambda key: (type(key).__name__, str(key)))
                ],
            ]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [description(part) for part in item]]
        require(item is None or type(item) in (bool, str, int, float), "Unsafe checkpoint metadata type")
        if type(item) is float:
            require(math.isfinite(item), "Nonfinite checkpoint scalar")
        return [type(item).__name__, item]

    return hashlib.sha256(
        json.dumps(description(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TrainingSettings:
    hidden_size: int = 64
    dt: float = 0.02
    residual_reward: bool = True
    noise_std: float = 0.05
    train_episodes: int = 768
    steps: int = 50
    epochs: int = 48
    batch_size: int = 32
    learning_rate: float = 0.001
    gradient_clip: float = 10.0
    rollout_horizon: int = 5
    rollout_weight: float = 0.5
    reward_scale: float = 4.0
    kl_weight: float = 0.01
    kl_balance: float = 0.8
    free_nats: float = 1.0
    auxiliary_horizons: tuple[int, ...] = (1, 3, 7)
    ema_momentum: float = 0.99
    variance_weight: float = 0.0
    covariance_weight: float = 0.0

    def __post_init__(self):
        for key in ("hidden_size", "train_episodes", "steps", "epochs", "batch_size", "rollout_horizon"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"Invalid {key}")
        for key in ("dt", "learning_rate", "gradient_clip"):
            require(finite(getattr(self, key), key) > 0, f"Invalid {key}")
        for key in (
            "noise_std",
            "rollout_weight",
            "reward_scale",
            "kl_weight",
            "free_nats",
            "variance_weight",
            "covariance_weight",
        ):
            finite(getattr(self, key), key)
        require(
            type(self.residual_reward) is bool and self.residual_reward,
            "The ablation requires the known-cost residual",
        )
        require(
            0 <= finite(self.kl_balance, "kl_balance") <= 1
            and 0 <= finite(self.ema_momentum, "ema_momentum") <= 1,
            "Invalid balance/momentum",
        )
        require(
            isinstance(self.auxiliary_horizons, tuple)
            and bool(self.auxiliary_horizons)
            and all(type(h) is int and h > 0 for h in self.auxiliary_horizons)
            and tuple(sorted(set(self.auxiliary_horizons))) == self.auxiliary_horizons,
            "Invalid auxiliary horizons",
        )
        require(
            self.variance_weight == self.covariance_weight == 0,
            "This ablation has zero regularization weights",
        )

    def configuration(self):
        result = asdict(self)
        result["auxiliary_horizons"] = list(self.auxiliary_horizons)
        result.update(
            optimizer=copy.deepcopy(ADAM),
            calibration=copy.deepcopy(CALIBRATION),
            dtype="torch.float32",
            device="cpu",
            model_class="GRUResidualRewardWorldModel",
        )
        return result

    @classmethod
    def from_plan(cls, plan):
        require(
            plan["optimizer"] == ADAM and plan["calibration"] == CALIBRATION,
            "Optimizer/calibration settings changed",
        )
        values = {field.name: plan[field.name] for field in fields(cls)}
        values["auxiliary_horizons"] = tuple(values["auxiliary_horizons"])
        return cls(**values)

    @classmethod
    def from_configuration(cls, config):
        values = {field.name: config[field.name] for field in fields(cls)}
        values["auxiliary_horizons"] = tuple(values["auxiliary_horizons"])
        result = cls(**values)
        require(result.configuration() == config, "Complete training configuration mismatch")
        return result

    def anchor_kwargs(self):
        return {
            key: getattr(self, key)
            for key in (
                "rollout_horizon",
                "rollout_weight",
                "reward_scale",
                "kl_weight",
                "kl_balance",
                "free_nats",
            )
        }


def _seed(seed):
    require(type(seed) is int and 0 <= seed < 2**32, "An explicit uint32 role seed is required")
    return seed


def _new_model(settings):
    return (
        GRUResidualRewardWorldModel(
            hidden_size=settings.hidden_size,
            dt=settings.dt,
            noise_std=settings.noise_std,
            residual_reward=settings.residual_reward,
        )
        .cpu()
        .float()
    )


def buffers(model):
    return {name: value for name, value in model.named_buffers()}


def _load_tensors(model, state, saved_buffers):
    expected = model.state_dict()
    require(
        set(state) == set(expected) and set(saved_buffers) == set(buffers(model)),
        "Checkpoint tensor membership",
    )
    for name, value in state.items():
        require(
            isinstance(value, Tensor)
            and value.shape == expected[name].shape
            and value.dtype == expected[name].dtype
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            "Checkpoint tensor schema/finiteness",
        )
    for name, value in saved_buffers.items():
        original = buffers(model)[name]
        require(
            isinstance(value, Tensor)
            and value.shape == original.shape
            and value.dtype == original.dtype
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            "Checkpoint buffer schema",
        )
        require(
            name not in state or torch.equal(value, state[name]),
            "Persistent buffer disagrees with state_dict",
        )
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        for name, value in saved_buffers.items():
            buffers(model)[name].copy_(value)


def make_initialization(student_seed: int, predictor_seed: int, settings: TrainingSettings | None = None):
    settings = settings or TrainingSettings()
    start = time.perf_counter()
    student_seed, predictor_seed = _seed(student_seed), _seed(predictor_seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(student_seed)
        student_before = torch.get_rng_state().clone()
        model = _new_model(settings)
        student_after = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(predictor_seed)
        predictor_before = torch.get_rng_state().clone()
        predictor = nn.Linear(settings.hidden_size, settings.hidden_size).cpu().float()
        predictor_after = torch.get_rng_state().clone()
    result = {
        "version": VERSION,
        "settings": settings.configuration(),
        "seeds": {"student": student_seed, "predictor": predictor_seed},
        "student_state": clone(model.state_dict()),
        "predictor_state": clone(predictor.state_dict()),
        "student_buffers": clone(buffers(model)),
        "rng_states": {
            "student_before": student_before,
            "student_after": student_after,
            "predictor_before": predictor_before,
            "predictor_after": predictor_after,
        },
    }
    result["hashes"] = {
        key: canonical_tensor_hash(result[key])
        for key in ("student_state", "predictor_state", "student_buffers", "rng_states")
    }
    result["initialization_wall_seconds"] = time.perf_counter() - start
    return result


def _validate_initialization(initialization, settings):
    require(
        set(initialization)
        == {
            "version",
            "settings",
            "seeds",
            "student_state",
            "predictor_state",
            "student_buffers",
            "rng_states",
            "hashes",
            "initialization_wall_seconds",
        },
        "Initialization schema",
    )
    require(
        initialization["version"] == VERSION and initialization["settings"] == settings.configuration(),
        "Initialization configuration",
    )
    require(set(initialization["seeds"]) == {"student", "predictor"}, "Initialization seed roles")
    for seed in initialization["seeds"].values():
        _seed(seed)
    require(
        set(initialization["hashes"])
        == {"student_state", "predictor_state", "student_buffers", "rng_states"},
        "Initialization hash membership",
    )
    for key, digest in initialization["hashes"].items():
        require(canonical_tensor_hash(initialization[key]) == digest, "Initialization tensor hash")
    require(
        set(initialization["rng_states"])
        == {"student_before", "student_after", "predictor_before", "predictor_after"},
        "Initialization RNG state membership",
    )
    for component in ("student", "predictor"):
        expected = torch.Generator(device="cpu").manual_seed(initialization["seeds"][component]).get_state()
        before, after = (initialization["rng_states"][component + suffix] for suffix in ("_before", "_after"))
        require(
            torch.equal(before, expected)
            and after.dtype == torch.uint8
            and after.device.type == "cpu"
            and after.shape == expected.shape,
            "Initialization RNG role/schema",
        )
    finite(initialization["initialization_wall_seconds"], "initialization time")


def backbone_parameters(model):
    values = {name: value for name, value in model.named_parameters() if name.startswith(BACKBONE_PREFIXES)}
    require(
        len(values) == 8
        and all(any(name.startswith(prefix) for name in values) for prefix in BACKBONE_PREFIXES),
        "Expected two GRUCell backbones with exactly eight named tensors",
    )
    return values


def _norm(values):
    # float64 descriptive measurement, separate from unchanged native clip_grad_norm_.
    terms = [value.detach().double().square().sum() for value in values if value is not None]
    return math.sqrt(float(torch.stack(terms).sum())) if terms else 0.0


def _check_batch(settings, packets, commands, rewards, transition_valid=None):
    require(all(isinstance(value, Tensor) for value in (packets, commands, rewards)), "Tensor batch required")
    batch = commands.shape[0] if isinstance(commands, Tensor) and commands.ndim == 3 else 0
    require(
        batch > 0
        and commands.shape == (batch, settings.steps, 2)
        and packets.shape == (batch, settings.steps + 1, 8)
        and rewards.shape == (batch, settings.steps),
        "Complete episode batch shapes",
    )
    for value in (packets, commands, rewards):
        require(
            isinstance(value, Tensor)
            and value.dtype == torch.float32
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            "Finite CPU float32 public/label tensors required",
        )
    require(
        bool(((packets[..., 6] == 0) | (packets[..., 6] == 1)).all()) and bool((packets[..., 7] >= 0).all()),
        "Public validity/age",
    )
    if transition_valid is not None:
        require(
            transition_valid.shape == commands.shape[:2]
            and transition_valid.dtype == torch.bool
            and transition_valid.device.type == "cpu"
            and bool(transition_valid.all()),
            "The unchanged anchor requires complete episodes; padding masks are not supported",
        )
    return batch


def anchor_work(settings, batch):
    steps, horizon = settings.steps, settings.rollout_horizon
    starts = max(0, steps - horizon + 1)
    rollout = horizon if settings.rollout_weight and starts else 0
    calls = {
        "student_assimilate": steps,
        "student_prefix_advance": steps,
        "student_rollout_advance": rollout,
        "student_posterior_kl": steps,
    }
    samples = {key: value * batch for key, value in calls.items()}
    samples["student_rollout_advance"] *= starts
    return {"batch_forward_calls": calls, "sample_forward_evaluations": samples, "counts_are_not_flops": True}


class ObjectiveTrainer:
    def __init__(
        self, arm, initialization, minibatch_seed, loss_multiplier, settings, source_sha256, runtime
    ):
        start = time.perf_counter()
        require(arm in ARMS, "Unknown objective arm")
        _validate_initialization(initialization, settings)
        self.arm, self.settings = arm, settings
        self.loss_multiplier = finite(loss_multiplier, "loss multiplier")
        require(arm != "anchor" or self.loss_multiplier == 0, "Anchor must have zero auxiliary multiplier")
        self.initialization = clone(initialization)
        self.source_sha256, self.runtime = clone(source_sha256), clone(runtime)
        require(
            isinstance(source_sha256, dict) and isinstance(runtime, dict), "Source/runtime mappings required"
        )
        self.minibatch_seed = _seed(minibatch_seed)
        self.minibatch_generator = torch.Generator(device="cpu").manual_seed(minibatch_seed)
        with torch.random.fork_rng(devices=[]):
            # Temporary construction is isolated and overwritten by exact paired tensors.
            torch.manual_seed(0)
            self.student = _new_model(settings)
            _load_tensors(self.student, initialization["student_state"], initialization["student_buffers"])
            if arm == "raw":
                self.auxiliary = RawEndpointAuxiliary(self.student, horizons=settings.auxiliary_horizons)
            elif arm == "latent":
                self.auxiliary = LatentConsistencyAuxiliary(
                    self.student,
                    horizons=settings.auxiliary_horizons,
                    momentum=settings.ema_momentum,
                    variance_weight=settings.variance_weight,
                    covariance_weight=settings.covariance_weight,
                )
            else:
                self.auxiliary = None
            if self.auxiliary is not None:
                _load_tensors(self.auxiliary.predictor, initialization["predictor_state"], {})
        self.student.train()
        if self.auxiliary is not None:
            self.auxiliary.train()
        self.named_trainable = {"student." + name: value for name, value in self.student.named_parameters()}
        if self.auxiliary is not None:
            self.named_trainable.update(
                {"predictor." + name: value for name, value in self.auxiliary.predictor.named_parameters()}
            )
        require(
            len({id(value) for value in self.named_trainable.values()}) == len(self.named_trainable),
            "Duplicate optimizer parameter",
        )
        self.optimizer_group_names = [list(self.named_trainable)]
        kwargs = {key: value for key, value in ADAM.items() if key != "name"}
        kwargs["betas"] = tuple(kwargs["betas"])
        self.optimizer = torch.optim.Adam(
            list(self.named_trainable.values()), lr=settings.learning_rate, **kwargs
        )
        self.successful_updates = self.optimizer_steps = self.ema_updates = 0
        self.failed = False
        self.cursor = {"epoch": 0, "batch": 0}
        self.log_chain_sha256 = hashlib.sha256(b"").hexdigest()
        self.setup_wall_seconds = time.perf_counter() - start
        self.training_wall_seconds = 0.0
        self.ema_witness = None
        self._check_models()

    def _check_models(self):
        models = [self.student] + ([self.auxiliary.teacher] if self.arm == "latent" else [])
        for model in models:
            require(
                type(model) is GRUResidualRewardWorldModel
                and model.hidden_size == self.settings.hidden_size
                and model.dt == self.settings.dt
                and model.noise_std == self.settings.noise_std
                and model.residual_reward is self.settings.residual_reward,
                "Student/teacher scalar configuration drift",
            )
            require(
                all(
                    value.device.type == "cpu" and value.dtype == torch.float32
                    for value in model.parameters()
                ),
                "Model must remain CPU float32",
            )
        if self.arm == "latent":
            require(
                not self.auxiliary.teacher.training
                and not any(p.requires_grad for p in self.auxiliary.teacher.parameters()),
                "EMA teacher must remain frozen/eval",
            )
            teacher_ids = {id(p) for p in self.auxiliary.teacher.parameters()}
            require(
                not teacher_ids.intersection(
                    id(p) for group in self.optimizer.param_groups for p in group["params"]
                ),
                "Teacher entered optimizer",
            )
        require(
            [[id(p) for p in group["params"]] for group in self.optimizer.param_groups]
            == [[id(p) for p in self.named_trainable.values()]],
            "Optimizer parameter membership/order changed",
        )
        actual = self.optimizer.state_dict()["param_groups"][0]
        expected = {key: value for key, value in ADAM.items() if key != "name"}
        expected.update(lr=self.settings.learning_rate, betas=tuple(ADAM["betas"]))
        require(set(actual) == set(expected) | {"params"}, "Optimizer scalar membership changed")
        require(
            all(actual[key] == value for key, value in expected.items()),
            "Optimizer scalar configuration drift",
        )

    def next_permutation(self):
        require(not self.failed, "Failed trainer is terminal")
        return torch.randperm(self.settings.train_episodes, generator=self.minibatch_generator)

    def train_batch(
        self,
        packets,
        commands,
        rewards,
        *,
        epoch,
        batch,
        indices=None,
        transition_valid=None,
        deadline_check: Callable[[], None] | None = None,
        capture_ema_witness=False,
    ):
        start = time.perf_counter()
        require(not self.failed, "Failed trainer is terminal")
        self.ema_witness = None
        check = deadline_check or (lambda: None)
        try:
            check()
            self._check_models()
            size = _check_batch(self.settings, packets, commands, rewards, transition_valid)
            require(
                type(epoch) is int and epoch >= 0 and type(batch) is int and batch >= 0,
                "Invalid epoch/batch cursor",
            )
            if indices is not None:
                require(
                    isinstance(indices, Tensor)
                    and indices.dtype == torch.int64
                    and indices.device.type == "cpu"
                    and indices.shape == (size,)
                    and len(set(indices.tolist())) == size
                    and bool(((indices >= 0) & (indices < self.settings.train_episodes)).all()),
                    "Batch index membership",
                )
            self.optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            anchor, anchor_metrics = sequence_loss(
                self.student, packets, commands, rewards, **self.settings.anchor_kwargs()
            )
            anchor_seconds = time.perf_counter() - tick
            check()
            auxiliary, auxiliary_metrics = anchor * 0, None
            tick = time.perf_counter()
            if self.auxiliary is not None:
                auxiliary, auxiliary_metrics = self.auxiliary(
                    self.student, packets, commands, transition_valid=transition_valid
                )
            auxiliary_seconds = time.perf_counter() - tick
            # A zero multiplier leaves the exact anchor graph/update unchanged,
            # while auxiliary computation is still honestly counted and charged.
            total = anchor + self.loss_multiplier * auxiliary if self.loss_multiplier else anchor
            require(
                bool(torch.isfinite(total)) and bool(torch.isfinite(auxiliary)),
                "Nonfinite training objective",
            )
            check()
            tick = time.perf_counter()
            total.backward()
            backward_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            backbone_norm = _norm(p.grad for p in backbone_parameters(self.student).values())
            predictor_norm = (
                _norm(p.grad for p in self.auxiliary.predictor.parameters())
                if self.auxiliary is not None
                else 0.0
            )
            norm = torch.nn.utils.clip_grad_norm_(
                list(self.named_trainable.values()), self.settings.gradient_clip, error_if_nonfinite=True
            )
            gradient_seconds = time.perf_counter() - tick
            check()
            tick = time.perf_counter()
            before = self._teacher_witness_state() if capture_ema_witness and self.arm == "latent" else None
            before_witness_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            self.optimizer.step()
            optimizer_seconds = time.perf_counter() - tick
            self.optimizer_steps += 1
            # No deadline check between a successful Adam step and its one EMA.
            tick = time.perf_counter()
            student_after = self._student_witness_state() if before is not None else None
            if self.arm == "latent":
                self._check_models()
                self.auxiliary.update_teacher(self.student)
                self.ema_updates += 1
            if before is not None:
                self.ema_witness = {
                    "momentum": self.settings.ema_momentum,
                    "update": self.optimizer_steps,
                    "teacher_before": before,
                    "student_after_optimizer": student_after,
                    "teacher_after": self._teacher_witness_state(),
                }
                self.ema_witness["sha256"] = canonical_state_hash(self.ema_witness)
            ema_seconds = time.perf_counter() - tick + before_witness_seconds
            self.successful_updates += 1
            self.cursor = {"epoch": epoch, "batch": batch}
            metrics = {
                "arm": self.arm,
                "epoch": epoch,
                "batch": batch,
                "update": self.successful_updates,
                "optimizer_steps": self.optimizer_steps,
                "ema_updates": self.ema_updates,
                "indices": indices.tolist() if indices is not None else None,
                "indices_sha256": canonical_tensor_hash({"indices": indices})
                if indices is not None
                else None,
                "anchor": anchor_metrics,
                "auxiliary": auxiliary_metrics,
                "loss_multiplier": self.loss_multiplier,
                "anchor_loss": float(anchor.detach()),
                "auxiliary_loss": float(auxiliary.detach()),
                "total_loss": float(total.detach()),
                "backbone_gradient_norm": backbone_norm,
                "predictor_gradient_norm": predictor_norm,
                "combined_gradient_norm": float(norm),
                "gradient_clipped": bool(norm > self.settings.gradient_clip),
                "anchor_work": anchor_work(self.settings, size),
                "auxiliary_work": None
                if auxiliary_metrics is None
                else {
                    "batch_forward_calls": auxiliary_metrics["batch_forward_calls"],
                    "sample_forward_evaluations": {
                        key: value * size for key, value in auxiliary_metrics["batch_forward_calls"].items()
                    },
                    "collapse_diagnostics_included_in_auxiliary_wall": self.arm == "latent",
                },
                "ema_witness_captured": before is not None,
                "costs": {
                    "anchor_forward_seconds": anchor_seconds,
                    "auxiliary_forward_seconds": auxiliary_seconds,
                    "backward_seconds": backward_seconds,
                    "gradient_norm_and_clip_seconds": gradient_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "ema_and_witness_seconds": ema_seconds,
                },
            }
            metrics["costs"]["batch_wall_seconds"] = time.perf_counter() - start
            previous = self.log_chain_sha256
            encoded = json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self.log_chain_sha256 = hashlib.sha256(bytes.fromhex(previous) + encoded).hexdigest()
            metrics.update(previous_log_sha256=previous, log_sha256=self.log_chain_sha256)
            check()
            return metrics
        except BaseException:
            self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.training_wall_seconds += time.perf_counter() - start

    def _student_witness_state(self):
        return {
            "parameters": clone(dict(self.student.named_parameters())),
            "buffers": clone(buffers(self.student)),
        }

    def _teacher_witness_state(self):
        return {
            "parameters": clone(dict(self.auxiliary.teacher.named_parameters())),
            "buffers": clone(buffers(self.auxiliary.teacher)),
        }

    def export_checkpoint(self, cursor=None):
        self._check_models()
        if cursor is not None:
            require(
                set(cursor) == {"epoch", "batch"} and all(type(v) is int and v >= 0 for v in cursor.values()),
                "Checkpoint cursor",
            )
        payload = {
            "version": VERSION,
            "arm": self.arm,
            "settings": self.settings.configuration(),
            "loss_multiplier": self.loss_multiplier,
            "source_sha256": clone(self.source_sha256),
            "runtime": clone(self.runtime),
            "initialization": clone(self.initialization),
            "student_state": clone(self.student.state_dict()),
            "student_buffers": clone(buffers(self.student)),
            "auxiliary_state": clone(self.auxiliary.state_dict()) if self.auxiliary is not None else None,
            "auxiliary_buffers": clone(buffers(self.auxiliary)) if self.auxiliary is not None else None,
            "auxiliary_configuration": self.auxiliary_configuration(),
            "optimizer_state": clone(self.optimizer.state_dict()),
            "optimizer_group_names": clone(self.optimizer_group_names),
            "minibatch_seed": self.minibatch_seed,
            "minibatch_rng_state": self.minibatch_generator.get_state().clone(),
            "cursor": clone(cursor if cursor is not None else self.cursor),
            "successful_updates": self.successful_updates,
            "optimizer_steps": self.optimizer_steps,
            "ema_updates": self.ema_updates,
            "failed": self.failed,
            "log_chain_sha256": self.log_chain_sha256,
            "setup_wall_seconds": self.setup_wall_seconds,
            "training_wall_seconds": self.training_wall_seconds,
        }
        payload["integrity_sha256"] = canonical_state_hash(payload)
        return payload

    def auxiliary_configuration(self):
        if self.auxiliary is None:
            return None
        return {
            "kind": self.arm,
            "horizons": list(self.settings.auxiliary_horizons),
            "momentum": self.settings.ema_momentum if self.arm == "latent" else None,
            "variance_weight": self.settings.variance_weight,
            "covariance_weight": self.settings.covariance_weight,
            "model": {
                "hidden_size": self.student.hidden_size,
                "dt": self.student.dt,
                "noise_std": self.student.noise_std,
                "residual_reward": self.student.residual_reward,
            },
            "teacher_present": self.arm == "latent",
        }

    def save_weights(self, path):
        self._check_models()
        state = clone(self.student.state_dict())
        with Path(path).open("xb") as handle:
            torch.save(state, handle)
        return {
            "canonical_tensor_sha256": canonical_tensor_hash(state),
            "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }


def make_trainer(
    arm, initialization, minibatch_seed, loss_multiplier, settings=None, *, source_sha256=None, runtime=None
):
    settings = settings or TrainingSettings.from_configuration(initialization["settings"])
    return ObjectiveTrainer(
        arm, initialization, minibatch_seed, loss_multiplier, settings, source_sha256 or {}, runtime or {}
    )


def restore_checkpoint(
    payload, *, expected_settings: TrainingSettings, expected_source_sha256: dict, expected_runtime: dict
):
    start = time.perf_counter()
    require(isinstance(payload, dict) and "integrity_sha256" in payload, "Checkpoint schema")
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    require(canonical_state_hash(body) == payload["integrity_sha256"], "Checkpoint integrity hash")
    require(
        payload["version"] == VERSION
        and payload["settings"] == expected_settings.configuration()
        and payload["source_sha256"] == expected_source_sha256
        and payload["runtime"] == expected_runtime,
        "Checkpoint configuration/source/runtime",
    )
    require(
        payload["failed"] is False, "Failed partial attempt cannot be restored for continuation/evaluation"
    )
    trainer = make_trainer(
        payload["arm"],
        payload["initialization"],
        payload["minibatch_seed"],
        payload["loss_multiplier"],
        expected_settings,
        source_sha256=expected_source_sha256,
        runtime=expected_runtime,
    )
    require(set(payload) == set(trainer.export_checkpoint()), "Checkpoint exact member set")
    require(
        payload["optimizer_group_names"] == trainer.optimizer_group_names
        and payload["auxiliary_configuration"] == trainer.auxiliary_configuration(),
        "Checkpoint optimizer/auxiliary configuration",
    )
    _load_tensors(trainer.student, payload["student_state"], payload["student_buffers"])
    if trainer.auxiliary is not None:
        _load_tensors(trainer.auxiliary, payload["auxiliary_state"], payload["auxiliary_buffers"])
    else:
        require(
            payload["auxiliary_state"] is None and payload["auxiliary_buffers"] is None,
            "Unexpected anchor auxiliary",
        )
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.minibatch_generator.set_state(payload["minibatch_rng_state"])
    for key in ("successful_updates", "optimizer_steps", "ema_updates"):
        require(type(payload[key]) is int and payload[key] >= 0, "Checkpoint update count")
        setattr(trainer, key, payload[key])
    require(
        trainer.optimizer_steps == trainer.successful_updates
        and trainer.ema_updates == (trainer.successful_updates if trainer.arm == "latent" else 0),
        "Checkpoint step/EMA arithmetic",
    )
    require(
        set(payload["cursor"]) == {"epoch", "batch"}
        and all(type(v) is int and v >= 0 for v in payload["cursor"].values()),
        "Checkpoint cursor",
    )
    trainer.cursor = clone(payload["cursor"])
    require(
        isinstance(payload["log_chain_sha256"], str) and len(payload["log_chain_sha256"]) == 64,
        "Checkpoint log chain",
    )
    require(
        all(character in "0123456789abcdef" for character in payload["log_chain_sha256"]),
        "Checkpoint log chain encoding",
    )
    trainer.log_chain_sha256 = payload["log_chain_sha256"]
    for key in ("setup_wall_seconds", "training_wall_seconds"):
        setattr(trainer, key, finite(payload[key], key))
    trainer._check_models()
    # Loading optimizer state can silently coerce dtypes; require exact roundtrip.
    require(
        canonical_state_hash(trainer.optimizer.state_dict())
        == canonical_state_hash(payload["optimizer_state"]),
        "Optimizer state coercion",
    )
    state = trainer.optimizer.state_dict()
    require(set(state) == {"state", "param_groups"}, "Optimizer state members")
    expected_ids = list(range(len(trainer.named_trainable)))
    require(state["param_groups"][0]["params"] == expected_ids, "Optimizer parameter IDs")
    active = {
        index: parameter
        for index, (name, parameter) in enumerate(trainer.named_trainable.items())
        if trainer.successful_updates and (name.startswith("student.") or trainer.loss_multiplier > 0)
    }
    require(set(state["state"]) == set(active), "Optimizer state coverage")
    for index, parameter in active.items():
        slots = state["state"][index]
        require(set(slots) == {"step", "exp_avg", "exp_avg_sq"}, "Adam slot membership")
        step = slots["step"]
        require(
            isinstance(step, Tensor)
            and step.dtype == torch.float32
            and step.device.type == "cpu"
            and step.shape == ()
            and float(step) == trainer.successful_updates,
            "Adam step counter",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            require(
                slots[key].shape == parameter.shape
                and slots[key].dtype == parameter.dtype
                and slots[key].device.type == "cpu"
                and bool(torch.isfinite(slots[key]).all()),
                "Adam moment schema",
            )
        require(bool((slots["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")
    trainer.restoration_wall_seconds = time.perf_counter() - start
    return trainer


def calibrate(initializations, public_data, settings: TrainingSettings, *, deadline_check=None):
    """Training-only fixed first-128-row gradient-scale calibration.

    Returns JSON-safe `receipt` and a separate named `gradients` tensor mapping
    for saved numerical norm reconstruction. No optimizer or EMA update occurs.
    The auditor can verify vector-to-norm arithmetic, not their neural origin.
    """
    begin = time.perf_counter()
    require(len(initializations) == 3, "Calibration requires all three paired initializations")
    require(
        isinstance(public_data, Mapping) and set(public_data) == {"packets", "commands", "rewards"},
        "Calibration public data allowlist",
    )
    data = [public_data[key] for key in ("packets", "commands", "rewards")]
    require(
        len(data[0]) >= 128 and all(len(value) == len(data[0]) for value in data),
        "Calibration needs the first four complete B32 batches",
    )
    before_global = torch.get_rng_state().clone()
    before_initial = [canonical_state_hash(value) for value in initializations]
    check = deadline_check or (lambda: None)
    gradients, rows, norms = {}, [], {arm: [] for arm in ARMS}
    costs = {
        "setup_seconds": 0.0,
        "anchor_forward_seconds": 0.0,
        "raw_forward_seconds": 0.0,
        "latent_forward_seconds": 0.0,
        "gradient_seconds": 0.0,
    }
    parameter_names = None
    try:
        for pair, initialization in enumerate(initializations):
            check()
            tick = time.perf_counter()
            trainers = {
                arm: make_trainer(arm, initialization, 0, 0 if arm == "anchor" else 1, settings)
                for arm in ARMS
            }
            costs["setup_seconds"] += time.perf_counter() - tick
            for batch in range(4):
                indices = list(range(batch * 32, (batch + 1) * 32))
                batch_data = [value[batch * 32 : (batch + 1) * 32] for value in data]
                _check_batch(settings, *batch_data)
                values, work = {}, {}
                for arm, trainer in trainers.items():
                    check()
                    params = backbone_parameters(trainer.student)
                    if parameter_names is None:
                        parameter_names = list(params)
                    require(list(params) == parameter_names, "Calibration backbone membership changed")
                    tick = time.perf_counter()
                    if arm == "anchor":
                        loss, metrics = sequence_loss(
                            trainer.student, *batch_data, **settings.anchor_kwargs()
                        )
                        work[arm] = anchor_work(settings, 32)
                    else:
                        loss, metrics = trainer.auxiliary(trainer.student, batch_data[0], batch_data[1])
                        work[arm] = {
                            "batch_forward_calls": metrics["batch_forward_calls"],
                            "sample_forward_evaluations": {
                                key: value * 32 for key, value in metrics["batch_forward_calls"].items()
                            },
                            "collapse_diagnostics_included_in_forward_wall": arm == "latent",
                        }
                    costs[arm + "_forward_seconds"] += time.perf_counter() - tick
                    require(bool(torch.isfinite(loss)), "Nonfinite calibration loss")
                    tick = time.perf_counter()
                    computed = torch.autograd.grad(loss, tuple(params.values()), allow_unused=False)
                    costs["gradient_seconds"] += time.perf_counter() - tick
                    value = _norm(computed)
                    require(math.isfinite(value) and value > 0, "Zero/nonfinite calibration backbone norm")
                    prefix = f"pair{pair}/batch{batch}/{arm}/"
                    for name, tensor in zip(params, computed, strict=True):
                        require(bool(torch.isfinite(tensor).all()), "Nonfinite calibration gradient vector")
                        gradients[prefix + name] = clone(tensor)
                    norms[arm].append(value)
                    values[arm] = {
                        "loss": float(loss.detach()),
                        "backbone_norm": value,
                        "gradient_sha256": canonical_tensor_hash(
                            {name: gradients[prefix + name] for name in params}
                        ),
                    }
                    require(
                        all(p.grad is None for p in trainer.named_trainable.values()),
                        "Calibration retained parameter gradients",
                    )
                rows.append(
                    {"pair": pair, "batch": batch, "indices": indices, "losses": values, "work": work}
                )
            for trainer in trainers.values():
                require(
                    canonical_tensor_hash(trainer.student.state_dict())
                    == initialization["hashes"]["student_state"]
                    and canonical_tensor_hash(buffers(trainer.student))
                    == initialization["hashes"]["student_buffers"],
                    "Calibration mutated student",
                )
                if trainer.auxiliary is not None:
                    require(
                        canonical_tensor_hash(trainer.auxiliary.predictor.state_dict())
                        == initialization["hashes"]["predictor_state"],
                        "Calibration mutated predictor",
                    )
                if trainer.arm == "latent":
                    require(
                        canonical_tensor_hash(trainer.auxiliary.teacher.state_dict())
                        == initialization["hashes"]["student_state"]
                        and canonical_tensor_hash(buffers(trainer.auxiliary.teacher))
                        == initialization["hashes"]["student_buffers"],
                        "Calibration mutated teacher",
                    )
                require(
                    trainer.successful_updates == trainer.optimizer_steps == trainer.ema_updates == 0,
                    "Calibration performed updates",
                )
        multipliers = {}
        anchor_median = float(np.median(norms["anchor"]))
        for arm in ("raw", "latent"):
            auxiliary_median = float(np.median(norms[arm]))
            unbounded = 0.5 * anchor_median / auxiliary_median
            multipliers[arm] = {
                "value": float(np.clip(unbounded, 0.01, 100.0)),
                "unclipped": unbounded,
                "bound_hit": not 0.01 <= unbounded <= 100.0,
                "anchor_median": anchor_median,
                "auxiliary_median": auxiliary_median,
            }
        require(
            before_initial == [canonical_state_hash(value) for value in initializations],
            "Calibration changed initial artifacts",
        )
        require(torch.equal(before_global, torch.get_rng_state()), "Calibration consumed global RNG")
        costs["wall_seconds"] = time.perf_counter() - begin
        receipt = {
            "version": VERSION,
            "rule": copy.deepcopy(CALIBRATION),
            "batch_size": 32,
            "pairs": 3,
            "training_indices": list(range(128)),
            "rows": rows,
            "norms": norms,
            "multipliers": multipliers,
            "backbone_parameter_names": parameter_names,
            "initialization_hashes": [value["hashes"] for value in initializations],
            "initial_artifact_hashes_before": before_initial,
            "initial_artifact_hashes_after": before_initial,
            "gradients_sha256": canonical_tensor_hash(gradients),
            "optimizer_updates": 0,
            "ema_updates": 0,
            "global_rng_unchanged": True,
            "costs": costs,
        }
        return {"receipt": receipt, "gradients": gradients}
    finally:
        # Do not hide BaseException failures; restore caller RNG even on failure.
        torch.set_rng_state(before_global)
