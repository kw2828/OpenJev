"""Engineering-only anchor trainer for explicit real-history comparators.

No corpus, evaluation, planner, auxiliary loss or scored protocol lives here.
Every update calls the unchanged complete-episode sequence_loss. Its real
prefix calls assimilate once per packet, then advance with that issued action;
open-loop windows use advance only. Thus bounded-history phase semantics and
current-packet resets are exercised during training, not added after fitting.

Checkpoints bind actual classes because compatible GRU tensor names do not
identify assimilation semantics. Restore supports engineering state parity
and future explicitly authorized uses; it grants no failed-study resume right.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType

import torch

from openjev.research.reacher_bounded_history import (
    BoundedThreePacketGRUWorldModel,
    parameter_and_operation_accounting,
)
from openjev.research.reacher_objective_training import (
    ADAM,
    anchor_work,
    canonical_state_hash,
    canonical_tensor_hash,
    clone,
)
from openjev.research.reacher_observation_baseline import (
    CurrentObservationGRUWorldModel,
    FeedForwardObservationWorldModel,
    operation_counts,
)
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import sequence_loss

VERSION = "reacher-memory-training-engineering-v1"
REGISTRY = MappingProxyType(
    {
        "residual_gru": GRUResidualRewardWorldModel,
        "current_gru": CurrentObservationGRUWorldModel,
        "bounded_gru": BoundedThreePacketGRUWorldModel,
        "packet_mlp": FeedForwardObservationWorldModel,
    }
)
SEMANTICS = {
    "residual_gru": "persistent real-prefix GRU; missing observations preserve predicted recurrent state",
    "current_gru": "zero hidden before every real packet, including missing; recurrent imagined advances",
    "bounded_gru": "rebuild from final3 real packets and2 issued commands; explicit startup and real phases",
    "packet_mlp": "overwrite state with every real packet; action-conditioned imagined packets only",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, f"Invalid {label}")
    require(not positive or value > 0, f"Positive {label} required")


def _seed(value):
    require(type(value) is int and 0 <= value < 2**32, "Explicit uint32 role seed required")
    return value


def _digest(value):
    require(
        isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"),
        "SHA256 required",
    )
    return value


def _seal(value):
    value["integrity_sha256"] = canonical_state_hash(value)
    return value


def _unseal(value):
    require(isinstance(value, dict) and "integrity_sha256" in value, "Sealed payload required")
    body = {k: v for k, v in value.items() if k != "integrity_sha256"}
    require(canonical_state_hash(body) == value["integrity_sha256"], "Payload integrity")
    return body


@dataclass(frozen=True)
class MemoryTrainingSettings:
    hidden_size: int = 64
    mlp_width: int = 107
    dt: float = 0.02
    noise_std: float = 0.05
    residual_reward: bool = True
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

    def __post_init__(self):
        for key in (
            "hidden_size",
            "mlp_width",
            "train_episodes",
            "steps",
            "epochs",
            "batch_size",
            "rollout_horizon",
        ):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"Positive integer {key}")
        for key in ("dt", "learning_rate", "gradient_clip"):
            _number(getattr(self, key), key, positive=True)
        for key in ("noise_std", "rollout_weight", "reward_scale", "kl_weight", "kl_balance", "free_nats"):
            _number(getattr(self, key), key)
        require(self.kl_balance <= 1 and self.residual_reward is True, "Balance/residual configuration")

    @property
    def batches_per_epoch(self):
        return math.ceil(self.train_episodes / self.batch_size)

    def configuration(self):
        return {
            **asdict(self),
            "optimizer": clone(ADAM),
            "dtype": "torch.float32",
            "device": "cpu",
            "objective": "unchanged_sequence_loss",
            "engineering_only": True,
        }

    @classmethod
    def from_configuration(cls, config):
        require(isinstance(config, dict), "Settings mapping")
        require({f.name for f in fields(cls)} <= set(config), "Settings fields")
        result = cls(**{f.name: config[f.name] for f in fields(cls)})
        require(result.configuration() == config, "Complete settings configuration")
        return result

    def anchor_kwargs(self):
        return {
            k: getattr(self, k)
            for k in (
                "rollout_horizon",
                "rollout_weight",
                "reward_scale",
                "kl_weight",
                "kl_balance",
                "free_nats",
            )
        }


def model_configuration(kind, settings):
    require(kind in REGISTRY, "Unknown model kind")
    return {
        "kind": kind,
        "model_class": REGISTRY[kind].__name__,
        "width": settings.mlp_width if kind == "packet_mlp" else settings.hidden_size,
        "dt": settings.dt,
        "noise_std": settings.noise_std,
        "residual_reward": settings.residual_reward,
        "real_assimilation": SEMANTICS[kind],
        "imagined_rollout": "advance only; no imaginary observation is assimilated",
        "engineering_only": True,
        "integrated_study": False,
    }


def _construct(kind, settings, seed):
    require(kind in REGISTRY, "Unknown model kind")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed))
        width = (
            {"width": settings.mlp_width} if kind == "packet_mlp" else {"hidden_size": settings.hidden_size}
        )
        return (
            REGISTRY[kind](
                **width,
                dt=settings.dt,
                noise_std=settings.noise_std,
                residual_reward=settings.residual_reward,
            )
            .cpu()
            .float()
        )


def make_initialization(gru_seed, mlp_seed, settings=None, *, role_names=None):
    """Independent named streams; a single GRU payload is shared by all3 GRUs.

    Caller supplies prospective namespace-resolved seeds. No study seed is
    selected here. Engineering tests deliberately use only synthetic seed410.
    """
    started = time.perf_counter()
    settings = settings or MemoryTrainingSettings()
    roles = role_names or {"gru": "initialization/gru", "mlp": "initialization/mlp"}
    require(
        set(roles) == {"gru", "mlp"}
        and all(isinstance(x, str) and x for x in roles.values())
        and len(set(roles.values())) == 2,
        "Distinct named initialization roles",
    )
    seeds = {"gru": _seed(gru_seed), "mlp": _seed(mlp_seed)}
    states = {
        family: clone(_construct(kind, settings, seeds[family]).state_dict())
        for family, kind in (("gru", "residual_gru"), ("mlp", "packet_mlp"))
    }
    return _seal(
        {
            "version": VERSION,
            "settings": settings.configuration(),
            "roles": clone(roles),
            "seeds": seeds,
            "states": states,
            "tensor_hashes": {k: canonical_tensor_hash(v) for k, v in states.items()},
            "wall_seconds": time.perf_counter() - started,
        }
    )


def _validate_initialization(initialization, settings):
    body = _unseal(initialization)
    require(
        set(body) == {"version", "settings", "roles", "seeds", "states", "tensor_hashes", "wall_seconds"},
        "Initialization schema",
    )
    require(
        body["version"] == VERSION and body["settings"] == settings.configuration(), "Initialization settings"
    )
    require(
        all(set(body[k]) == {"gru", "mlp"} for k in ("roles", "seeds", "states", "tensor_hashes")),
        "Initialization family membership",
    )
    require(
        all(isinstance(x, str) and x for x in body["roles"].values())
        and len(set(body["roles"].values())) == 2,
        "Initialization roles",
    )
    _number(body["wall_seconds"], "initialization time")
    for family, kind in (("gru", "residual_gru"), ("mlp", "packet_mlp")):
        expected = _construct(kind, settings, body["seeds"][family]).state_dict()
        require(
            canonical_tensor_hash(body["states"][family])
            == body["tensor_hashes"][family]
            == canonical_tensor_hash(expected),
            "Exact seeded initialization",
        )


def make_orders(seed, settings, *, role_name="minibatches/paired"):
    require(isinstance(role_name, str) and role_name, "Named minibatch stream required")
    generator = torch.Generator(device="cpu").manual_seed(_seed(seed))
    initial = generator.get_state().clone()
    orders = torch.stack(
        [torch.randperm(settings.train_episodes, generator=generator) for _ in range(settings.epochs)]
    )
    return _seal(
        {
            "version": VERSION,
            "seed": seed,
            "role": role_name,
            "orders": orders,
            "initial_rng": initial,
            "final_rng": generator.get_state().clone(),
        }
    )


def _validate_orders(orders, settings):
    _unseal(orders)
    expected = make_orders(orders["seed"], settings, role_name=orders["role"])
    require(
        canonical_state_hash(orders) == canonical_state_hash(expected),
        "Exact complete paired minibatch order",
    )


def _public_data(data, settings):
    require(
        isinstance(data, dict) and set(data) == {"packets", "commands", "rewards"},
        "Only public packets/issued commands/native reward labels",
    )
    n, t = settings.train_episodes, settings.steps
    for key, shape in {"packets": (n, t + 1, 8), "commands": (n, t, 2), "rewards": (n, t)}.items():
        value = data[key]
        require(
            isinstance(value, torch.Tensor)
            and value.shape == shape
            and value.dtype == torch.float32
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            f"Complete finite CPU float32 {key}",
        )
    packets = data["packets"]
    require(
        bool(((packets[..., 6] == 0) | (packets[..., 6] == 1)).all()) and bool((packets[..., 7] >= 0).all()),
        "Public validity/age",
    )
    require(bool((packets[:, 0, 6] == 1).all()), "Every complete episode starts with a real observation")
    require(
        bool((packets[..., :4][packets[..., 6] == 0] == 0).all()), "Missing angular fields must be masked"
    )
    require(bool((data["commands"].abs() <= 1).all()), "Issued commands clipped to [-1,1]")
    return canonical_tensor_hash(data)


def _load_weights(model, weights):
    expected = model.state_dict()
    require(isinstance(weights, dict) and set(weights) == set(expected), "Exact model tensor membership")
    for name, value in weights.items():
        require(
            isinstance(value, torch.Tensor)
            and value.shape == expected[name].shape
            and value.dtype == torch.float32
            and value.device.type == "cpu"
            and bool(torch.isfinite(value).all()),
            f"Model tensor schema: {name}",
        )
    model.load_state_dict(weights, strict=True)


class MemoryTrainer:
    def __init__(self, kind, initialization, orders, public_data, *, source_sha256=None, runtime=None):
        started = time.perf_counter()
        self.settings = MemoryTrainingSettings.from_configuration(initialization["settings"])
        _validate_initialization(initialization, self.settings)
        _validate_orders(orders, self.settings)
        self.kind = kind
        self.model_configuration = model_configuration(kind, self.settings)
        self.initialization, self.orders = clone(initialization), clone(orders)
        self._orders_state_sha256 = canonical_state_hash(self.orders)
        self._data = clone(public_data)
        self.data_sha256 = _public_data(self._data, self.settings)
        self.source_sha256, self.runtime = clone(source_sha256 or {}), clone(runtime or {})
        require(
            isinstance(self.source_sha256, dict) and isinstance(self.runtime, dict), "Source/runtime mappings"
        )
        for digest in self.source_sha256.values():
            _digest(digest)
        canonical_state_hash(self.runtime)
        family = "mlp" if kind == "packet_mlp" else "gru"
        self.student = _construct(kind, self.settings, initialization["seeds"][family])
        _load_weights(self.student, initialization["states"][family])
        self.student.train()
        self.named_trainable = dict(self.student.named_parameters())
        self.optimizer_group_names = [list(self.named_trainable)]
        kwargs = {k: v for k, v in ADAM.items() if k != "name"}
        kwargs["betas"] = tuple(kwargs["betas"])
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.settings.learning_rate, **kwargs)
        self.successful_updates = self.optimizer_steps = 0
        self.failed = False
        self.log_chain_sha256 = canonical_state_hash([])
        self.training_wall_seconds = self.restoration_wall_seconds = 0.0
        self.setup_wall_seconds = time.perf_counter() - started
        self._check_live()

    @property
    def cursor(self):
        epoch, batch = divmod(self.successful_updates, self.settings.batches_per_epoch)
        return {"epoch": epoch, "batch": batch}

    def next_indices(self):
        require(not self.failed, "Failed trainer is terminal")
        epoch, batch = self.cursor.values()
        require(epoch < self.settings.epochs, "All scheduled updates completed")
        start = batch * self.settings.batch_size
        return self.orders["orders"][epoch, start : start + self.settings.batch_size].clone()

    def _check_live(self):
        s, model = self.settings, self.student
        require(
            type(model) is REGISTRY[self.kind]
            and model.dt == s.dt
            and model.noise_std == s.noise_std
            and model.residual_reward is s.residual_reward,
            "Actual model class/scalar configuration drift",
        )
        width = model.width if self.kind == "packet_mlp" else model.hidden_size
        require(
            width == self.model_configuration["width"]
            and self.model_configuration == model_configuration(self.kind, s),
            "Model width/semantics drift",
        )
        require(
            model.training and not list(model.named_buffers()), "Expected train-mode model with no buffers"
        )
        require(
            [(n, id(p)) for n, p in model.named_parameters()]
            == [(n, id(p)) for n, p in self.named_trainable.items()],
            "Named parameter ownership",
        )
        require(
            all(
                p.dtype == torch.float32
                and p.device.type == "cpu"
                and p.requires_grad
                and bool(torch.isfinite(p).all())
                for p in model.parameters()
            ),
            "Finite trainable CPU float32 weights",
        )
        require(
            len({id(p) for p in model.parameters()}) == len(self.named_trainable),
            "Duplicate trainable parameter",
        )
        require(
            [[id(p) for p in g["params"]] for g in self.optimizer.param_groups]
            == [[id(p) for p in self.named_trainable.values()]],
            "Optimizer ownership/order",
        )
        expected = {k: v for k, v in ADAM.items() if k != "name"}
        expected.update(lr=s.learning_rate, betas=tuple(ADAM["betas"]))
        actual = self.optimizer.state_dict()["param_groups"][0]
        require(
            set(actual) == set(expected) | {"params"} and all(actual[k] == v for k, v in expected.items()),
            "Adam recipe drift",
        )

    def _work(self, batch):
        calls = anchor_work(self.settings, batch)
        samples = calls["sample_forward_evaluations"]
        a = samples["student_assimilate"]
        d = samples["student_prefix_advance"] + samples["student_rollout_advance"]
        if self.kind == "bounded_gru":
            operations = parameter_and_operation_accounting(
                self.student, assimilate_samples=a, advance_samples=d
            )
        elif self.kind in ("current_gru", "packet_mlp"):
            operations = operation_counts(self.student, assimilate_samples=a, advance_samples=d)
        else:
            h = self.settings.hidden_size
            operations = {
                "gru_cell_sample_calls": a + d,
                "linear_layer_sample_calls": 4 * d,
                "analytic_reward_sample_calls": d,
                "dense_affine_macs": a * 3 * h * (h + 8) + d * (5 * h * h + 25 * h),
                "counts_are_not_total_flops_or_measured_wall_time": True,
            }
        return {"interfaces": calls, "operations": operations, "compute_matched": False}

    def train_next(self, *, deadline_check=None):
        started = time.perf_counter()
        require(not self.failed, "Failed trainer is terminal")
        check = deadline_check or (lambda: None)
        try:
            check()
            self._check_live()
            require(_public_data(self._data, self.settings) == self.data_sha256, "Training data changed")
            require(canonical_state_hash(self.orders) == self._orders_state_sha256, "Minibatch order changed")
            cursor, indices = self.cursor, self.next_indices()
            batch = [self._data[k][indices] for k in ("packets", "commands", "rewards")]
            self.optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            loss, values = sequence_loss(self.student, *batch, **self.settings.anchor_kwargs())
            forward_seconds = time.perf_counter() - tick
            require(bool(torch.isfinite(loss)), "Nonfinite anchor loss")
            check()
            tick = time.perf_counter()
            loss.backward()
            backward_seconds = time.perf_counter() - tick
            require(all(p.grad is not None for p in self.student.parameters()), "Missing trainable gradient")
            tick = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.settings.gradient_clip, error_if_nonfinite=True
            )
            gradient_seconds = time.perf_counter() - tick
            check()
            tick = time.perf_counter()
            self.optimizer.step()
            optimizer_seconds = time.perf_counter() - tick
            self.optimizer_steps += 1
            self.successful_updates += 1
            self._check_live()
            row = {
                "kind": self.kind,
                **cursor,
                "update": self.successful_updates,
                "optimizer_steps": self.optimizer_steps,
                "indices": indices.tolist(),
                "indices_sha256": canonical_tensor_hash({"indices": indices}),
                "anchor": values,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "gradient_clipped": bool(norm > self.settings.gradient_clip),
                "work": self._work(len(indices)),
                "costs": {
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "gradient_clip_seconds": gradient_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "batch_wall_seconds": time.perf_counter() - started,
                },
                "previous_log_sha256": self.log_chain_sha256,
            }
            self.log_chain_sha256 = canonical_state_hash(row)
            row["log_sha256"] = self.log_chain_sha256
            check()
            return row
        except BaseException:
            self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.training_wall_seconds += time.perf_counter() - started

    def export_checkpoint(self):
        self._check_live()
        return _seal(
            {
                "version": VERSION,
                "kind": self.kind,
                "settings": self.settings.configuration(),
                "model_configuration": clone(self.model_configuration),
                "initialization": clone(self.initialization),
                "orders": clone(self.orders),
                "data_sha256": self.data_sha256,
                "source_sha256": clone(self.source_sha256),
                "runtime": clone(self.runtime),
                "student_state": clone(self.student.state_dict()),
                "optimizer_state": clone(self.optimizer.state_dict()),
                "optimizer_group_names": clone(self.optimizer_group_names),
                "successful_updates": self.successful_updates,
                "optimizer_steps": self.optimizer_steps,
                "cursor": self.cursor,
                "failed": self.failed,
                "log_chain_sha256": self.log_chain_sha256,
                "setup_wall_seconds": self.setup_wall_seconds,
                "training_wall_seconds": self.training_wall_seconds,
                "restoration_wall_seconds": self.restoration_wall_seconds,
            }
        )

    def save_checkpoint(self, path):
        with Path(path).open("xb") as handle:
            torch.save(self.export_checkpoint(), handle)


def make_trainer(kind, initialization, orders, public_data, *, source_sha256=None, runtime=None):
    return MemoryTrainer(
        kind, initialization, orders, public_data, source_sha256=source_sha256, runtime=runtime
    )


def restore_checkpoint(
    payload,
    public_data,
    *,
    expected_kind,
    expected_settings,
    expected_initialization_sha256,
    expected_orders_sha256,
    expected_data_sha256,
    expected_source_sha256,
    expected_runtime,
):
    """Restore actual architecture, exact Adam tensors and next saved minibatch.

    External expected bindings are mandatory. The embedded integrity digest is
    a corruption check, not an independent provenance authority.
    """
    started = time.perf_counter()
    _unseal(payload)
    require(
        payload["version"] == VERSION
        and payload["kind"] == expected_kind
        and payload["settings"] == expected_settings.configuration(),
        "Expected actual model kind/settings",
    )
    require(
        payload["source_sha256"] == expected_source_sha256 and payload["runtime"] == expected_runtime,
        "Expected source/runtime",
    )
    require(
        payload["initialization"]["integrity_sha256"] == _digest(expected_initialization_sha256)
        and payload["orders"]["integrity_sha256"] == _digest(expected_orders_sha256)
        and payload["data_sha256"] == _digest(expected_data_sha256),
        "Expected initialization/order/data",
    )
    require(payload["failed"] is False, "Failed partial trainer is terminal")
    result = make_trainer(
        expected_kind,
        payload["initialization"],
        payload["orders"],
        public_data,
        source_sha256=expected_source_sha256,
        runtime=expected_runtime,
    )
    require(set(payload) == set(result.export_checkpoint()), "Exact checkpoint membership")
    require(
        result.data_sha256 == expected_data_sha256
        and payload["model_configuration"] == result.model_configuration
        and payload["optimizer_group_names"] == result.optimizer_group_names,
        "Model semantics/data/optimizer names",
    )
    for key in ("successful_updates", "optimizer_steps"):
        require(
            type(payload[key]) is int
            and 0 <= payload[key] <= expected_settings.epochs * expected_settings.batches_per_epoch,
            "Update count bounds",
        )
        setattr(result, key, payload[key])
    require(
        result.successful_updates == result.optimizer_steps and payload["cursor"] == result.cursor,
        "Exact next order cursor",
    )
    _load_weights(result.student, payload["student_state"])
    state = payload["optimizer_state"]
    require(isinstance(state, dict) and set(state) == {"state", "param_groups"}, "Optimizer state schema")
    count = len(result.named_trainable)
    require(
        len(state["param_groups"]) == 1 and state["param_groups"][0]["params"] == list(range(count)),
        "Optimizer parameter ID order",
    )
    require(
        set(state["state"]) == (set(range(count)) if result.successful_updates else set()),
        "Optimizer state coverage",
    )
    for index, parameter in enumerate(result.named_trainable.values()):
        if not result.successful_updates:
            break
        slots = state["state"][index]
        require(set(slots) == {"step", "exp_avg", "exp_avg_sq"}, "Exact Adam slots")
        for key, shape in (("step", ()), ("exp_avg", parameter.shape), ("exp_avg_sq", parameter.shape)):
            value = slots[key]
            require(
                isinstance(value, torch.Tensor)
                and value.shape == shape
                and value.dtype == torch.float32
                and value.device.type == "cpu"
                and bool(torch.isfinite(value).all()),
                "Finite CPU Adam tensor schema",
            )
        require(
            float(slots["step"]) == result.successful_updates and bool((slots["exp_avg_sq"] >= 0).all()),
            "Adam step/second moment",
        )
    result.optimizer.load_state_dict(clone(state))
    require(
        canonical_state_hash(result.optimizer.state_dict()) == canonical_state_hash(state),
        "No optimizer coercion",
    )
    result.log_chain_sha256 = _digest(payload["log_chain_sha256"])
    for key in ("setup_wall_seconds", "training_wall_seconds", "restoration_wall_seconds"):
        _number(payload[key], key)
        setattr(result, key, payload[key])
    result._check_live()
    result.restoration_wall_seconds += time.perf_counter() - started
    return result
