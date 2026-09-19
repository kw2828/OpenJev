"""Additive, prospective anchor trainer for the two-observation replay control.

Caller authentication supplies ORIGINAL paired initial tensors and the complete
saved minibatch order. This adapter does not select streams, authorize a study,
infer that supplied weights are initial, or admit old fitted checkpoints merely
because their tensor schema matches. External provenance remains mandatory.

Reuse only the existing anchor update/cursor/save conventions and pure checks;
do not change any frozen registry. Actual class, full reconstruction semantics,
data/order/initial hashes and Adam state are bound on every resume. There are no
neural random draws. Minibatch RNG states are deterministically linked to saved
orders and the update cursor; ambient CPU RNG is preserved, including failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from openjev.research.reacher_memory_training import (
    MemoryTrainer,
    MemoryTrainingSettings,
    _digest,
    _load_weights,
    _number,
    _public_data,
    _seal,
    _seed,
    _unseal,
    require,
)
from openjev.research.reacher_objective_training import (
    ADAM,
    anchor_work,
    canonical_state_hash,
    canonical_tensor_hash,
    clone,
)
from openjev.research.reacher_two_observation_history import (
    AGE_ATOL,
    AGE_RTOL,
    MAX_STEPS,
    WINDOW,
    TwoObservationHistoryGRUWorldModel,
    parameter_and_operation_accounting,
)

VERSION = "reacher-two-observation-training-v1"
KIND = "two_observation_gru"


@dataclass(frozen=True)
class TrainingSettings(MemoryTrainingSettings):
    """The existing anchor recipe; mlp_width is retained as unused paired metadata.

    Smaller episode counts/lengths are supported for synthetic tests. This is
    not a scored protocol and specifies no quality/continuation thresholds.
    """

    def __post_init__(self):
        super().__post_init__()
        require(self.steps <= MAX_STEPS, "Two-observation component supports at most fifty actions")

    def configuration(self):
        return {**super().configuration(), "adapter_version": VERSION,
                "model_class": TwoObservationHistoryGRUWorldModel.__name__,
                "mlp_width_usage": "unused; retained original paired settings metadata"}


def _construct(settings):
    # Construction is isolated, not a new training initialization stream.
    # Every tensor is overwritten by the externally authenticated initial state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        return TwoObservationHistoryGRUWorldModel(
            hidden_size=settings.hidden_size, dt=settings.dt,
            noise_std=settings.noise_std, residual_reward=settings.residual_reward,
        ).cpu().float()


def _orders(payload, settings):
    body = _unseal(payload)
    require(set(body) == {"version", "seed", "role", "orders", "initial_rng", "final_rng"},
            "Exact inherited order payload")
    require(isinstance(body["version"], str) and body["version"]
            and isinstance(body["role"], str) and body["role"], "Named inherited order identity")
    generator = torch.Generator(device="cpu").manual_seed(_seed(body["seed"]))
    states, expected = [generator.get_state().clone()], []
    for _ in range(settings.epochs):
        expected.append(torch.randperm(settings.train_episodes, generator=generator))
        states.append(generator.get_state().clone())
    for key, value in (("orders", torch.stack(expected)), ("initial_rng", states[0]), ("final_rng", states[-1])):
        supplied = body[key]
        require(isinstance(supplied, torch.Tensor) and supplied.device.type == "cpu"
                and supplied.dtype == value.dtype and supplied.shape == value.shape
                and torch.equal(supplied, value), "Exact seeded order/RNG: " + key)
    return states


def _data(data, settings):
    digest = _public_data(data, settings)
    packets = data["packets"]
    require(torch.equal(packets[..., 4:6], packets[:, :1, 4:6].expand(-1, settings.steps + 1, -1)),
            "Public static target changed")
    latest = torch.full((settings.train_episodes,), -1, dtype=torch.int64)
    older = latest.clone()
    for point in range(settings.steps + 1):
        visible = packets[:, point, 6] == 1
        older = torch.where(visible, latest, older)
        latest = torch.where(visible, point, latest)
        age = (point - latest).float() * settings.dt
        require(bool(torch.isclose(packets[:, point, 7], age, rtol=AGE_RTOL, atol=AGE_ATOL).all()),
                "Public age disagrees with observed integer clock")
        require(bool((packets[:, point, 7][visible] == 0).all()), "Visible measurement age must be zero")
        anchor = torch.where(older >= 0, older, latest)
        require(bool((point - anchor < WINDOW).all()), "Unsupported two-observation history overflow")
    return digest


def _metadata(provenance, sources, runtime):
    for name, value in (("provenance", provenance), ("source hashes", sources), ("runtime", runtime)):
        require(isinstance(value, dict) and bool(value) and all(isinstance(k, str) and k for k in value),
                "Explicit nonempty " + name)
        canonical_state_hash(value)
    for digest in sources.values():
        _digest(digest)


class TwoObservationTrainer(MemoryTrainer):
    """Reuse unchanged anchor update logic, overriding all exact-class admission.

    ``expected_initial_sha256`` hashes named initial tensors; data uses the same
    named-tensor hash. ``expected_orders_sha256`` hashes the entire supplied
    order payload, including its embedded integrity and RNG identities. Source,
    runtime and provenance are caller-authenticated bindings, not inferred truth.
    """

    def __init__(self, initial_weights, orders, public_data, *, settings,
                 expected_initial_sha256, expected_orders_sha256, expected_data_sha256,
                 provenance, source_sha256, runtime):
        started = time.perf_counter()
        ambient = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            require(type(settings) is TrainingSettings, "Actual two-observation training settings required")
            _metadata(provenance, source_sha256, runtime)
            self.settings, self.kind = settings, KIND
            self.provenance, self.source_sha256, self.runtime = clone(provenance), clone(source_sha256), clone(runtime)
            self.initialization = {"weights": clone(initial_weights), "tensor_sha256": _digest(expected_initial_sha256)}
            require(canonical_tensor_hash(initial_weights) == expected_initial_sha256, "Authenticated original initial tensors")
            require(canonical_state_hash(orders) == _digest(expected_orders_sha256), "Authenticated paired order")
            self.orders = clone(orders)
            self._order_rng_states = _orders(self.orders, settings)
            self._orders_state_sha256 = expected_orders_sha256
            self._data = clone(public_data)
            self.data_sha256 = _data(self._data, settings)
            require(self.data_sha256 == _digest(expected_data_sha256), "Authenticated public training data")
            self.student = _construct(settings)
            _load_weights(self.student, self.initialization["weights"])
            self.student.train()
            self.model_configuration = self.student.configuration()
            self.named_trainable = dict(self.student.named_parameters())
            self.optimizer_group_names = [list(self.named_trainable)]
            kwargs = {k: clone(v) for k, v in ADAM.items() if k != "name"}
            kwargs["betas"] = tuple(kwargs["betas"])
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=settings.learning_rate, **kwargs)
            self.successful_updates = self.optimizer_steps = 0
            self.failed, self.failure, self.last_attempt = False, None, None
            self.log_chain_sha256 = canonical_state_hash([])
            self.training_wall_seconds = self.restoration_wall_seconds = 0.0
            self._binding_sha256 = canonical_state_hash(self._binding())
            self._check_live()
            require(torch.equal(ambient, torch.get_rng_state()), "Unexpected setup random draws")
        self.setup_wall_seconds = time.perf_counter() - started

    def _binding(self):
        return {"kind": self.kind, "settings": self.settings.configuration(),
                "initialization": self.initialization, "provenance": self.provenance,
                "sources": self.source_sha256, "runtime": self.runtime,
                "data_sha256": self.data_sha256, "orders_sha256": self._orders_state_sha256,
                "order_rng_states_sha256": canonical_tensor_hash({str(i): state for i, state in enumerate(self._order_rng_states)})}

    def _check_live(self):
        model, settings = self.student, self.settings
        require(type(model) is TwoObservationHistoryGRUWorldModel
                and type(settings) is TrainingSettings and self.kind == KIND
                and model.hidden_size == settings.hidden_size and model.dt == settings.dt
                and model.noise_std == settings.noise_std and model.residual_reward is settings.residual_reward
                and model.configuration() == self.model_configuration, "Actual class/full history configuration drift")
        require(canonical_state_hash(self._binding()) == self._binding_sha256, "Bound initialization/settings/provenance drift")
        require(model.training and not list(model.named_buffers()), "Train-mode model without buffers required")
        require([(n, id(p)) for n, p in model.named_parameters()]
                == [(n, id(p)) for n, p in self.named_trainable.items()], "Named parameter ownership")
        require(self.optimizer_group_names == [list(self.named_trainable)], "Named optimizer group drift")
        require(all(p.dtype == torch.float32 and p.device.type == "cpu" and p.requires_grad
                    and bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite trainable CPU float32 weights")
        require(len({id(p) for p in model.parameters()}) == len(self.named_trainable), "Duplicate parameter")
        require(type(self.optimizer) is torch.optim.Adam
                and [[id(p) for p in g["params"]] for g in self.optimizer.param_groups]
                == [[id(p) for p in self.named_trainable.values()]], "Exact Adam ownership/order")
        expected = {k: v for k, v in ADAM.items() if k != "name"}
        expected.update(lr=settings.learning_rate, betas=tuple(ADAM["betas"]))
        actual = self.optimizer.state_dict()["param_groups"][0]
        require(set(actual) == set(expected) | {"params"}
                and all(actual[k] == v for k, v in expected.items()), "Unchanged Adam recipe")

    def _work(self, batch):
        calls = anchor_work(self.settings, batch)
        samples = calls["sample_forward_evaluations"]
        operations = parameter_and_operation_accounting(
            self.student, assimilate_samples=samples["student_assimilate"],
            advance_samples=samples["student_prefix_advance"] + samples["student_rollout_advance"],
        )
        return {"interfaces": calls, "operations": operations, "compute_matched": False}

    def rng_identity(self):
        consumed = (self.successful_updates + self.settings.batches_per_epoch - 1) // self.settings.batches_per_epoch
        return {"neural_rng": "no draws; deterministic CPU model; ambient RNG preserved",
                "order_role": self.orders["role"], "order_seed": self.orders["seed"],
                "consumed_epoch_permutations": consumed,
                "initial_rng": self._order_rng_states[0].clone(),
                "current_rng": self._order_rng_states[consumed].clone(),
                "final_rng": self._order_rng_states[-1].clone(),
                "convention": "immutable full orders; logical RNG advances once upon first batch of each epoch"}

    def train_next(self, *, deadline_check=None):
        require(not self.failed, "Failed trainer is terminal")
        started, previous_wall = time.perf_counter(), self.training_wall_seconds
        ambient = torch.get_rng_state().clone()
        counts = {"gru_cell_sample_calls": 0, "linear_layer_sample_calls": 0}
        handles, cursor = [], self.cursor

        def guard():
            require(torch.equal(ambient, torch.get_rng_state()), "Unexpected neural/guard random draws")
            if deadline_check is not None:
                deadline_check()
            require(torch.equal(ambient, torch.get_rng_state()), "Unexpected neural/guard random draws")

        def hook(key):
            def completed(_module, args, _output):
                counts[key] += len(args[0])
            return completed

        try:
            with torch.random.fork_rng(devices=[]):
                self._check_live()
                for module in self.student.modules():
                    if isinstance(module, nn.GRUCell):
                        handles.append(module.register_forward_hook(hook("gru_cell_sample_calls")))
                    elif isinstance(module, nn.Linear):
                        handles.append(module.register_forward_hook(hook("linear_layer_sample_calls")))
                # Parent method contains no class construction or registry access.
                # It dispatches _check_live/_work above and calls unchanged sequence_loss.
                row = super().train_next(deadline_check=guard)
                require(all(counts[k] == row["work"]["operations"][k] for k in counts), "Observed complete neural work")
                guard()
            self.last_attempt = {"status": "completed", "cursor": cursor, "completed_neural_sample_calls": counts,
                                 "wall_seconds": time.perf_counter() - started}
            row["observed_neural_sample_calls"] = clone(counts)
            row["rng_identity_sha256"] = canonical_state_hash(self.rng_identity())
            row["costs"]["batch_wall_seconds"] = self.last_attempt["wall_seconds"]
            row.pop("log_sha256")
            self.log_chain_sha256 = canonical_state_hash(row)
            row["log_sha256"] = self.log_chain_sha256
            return row
        except BaseException as error:
            self.failed = True
            self.failure = {"type": type(error).__name__, "message": str(error), "cursor_before": cursor,
                            "successful_updates": self.successful_updates, "optimizer_steps": self.optimizer_steps}
            self.last_attempt = {"status": "failed", "cursor": cursor, "completed_neural_sample_calls": counts,
                                 "wall_seconds": time.perf_counter() - started,
                                 "limit": "completed module calls only; interrupted internal work remains in wall time"}
            raise
        finally:
            for handle in handles:
                handle.remove()
            self.optimizer.zero_grad(set_to_none=True)
            self.training_wall_seconds = previous_wall + time.perf_counter() - started

    def export_checkpoint(self):
        self._check_live()
        require(_data(self._data, self.settings) == self.data_sha256, "Training data changed")
        require(canonical_state_hash(self.orders) == self._orders_state_sha256, "Minibatch order changed")
        return _seal({
            "version": VERSION, "kind": KIND, "settings": self.settings.configuration(),
            "model_configuration": clone(self.model_configuration), "initialization": clone(self.initialization),
            "orders": clone(self.orders), "data_sha256": self.data_sha256, "provenance": clone(self.provenance),
            "source_sha256": clone(self.source_sha256), "runtime": clone(self.runtime),
            "student_state": clone(self.student.state_dict()), "optimizer_state": clone(self.optimizer.state_dict()),
            "optimizer_group_names": clone(self.optimizer_group_names), "successful_updates": self.successful_updates,
            "optimizer_steps": self.optimizer_steps, "cursor": self.cursor, "rng": self.rng_identity(),
            "failed": self.failed, "failure": clone(self.failure), "last_attempt": clone(self.last_attempt),
            "log_chain_sha256": self.log_chain_sha256, "setup_wall_seconds": self.setup_wall_seconds,
            "training_wall_seconds": self.training_wall_seconds, "restoration_wall_seconds": self.restoration_wall_seconds,
        })


def restore_checkpoint(payload, public_data, *, expected_checkpoint_sha256, expected_kind, expected_settings,
                       expected_initial_sha256, expected_orders_sha256, expected_data_sha256,
                       expected_provenance, expected_source_sha256, expected_runtime):
    """Actual-class/Adam resume, never authority to resume a stopped scored run.

    The expected checkpoint digest is its canonical sealed-body digest, supplied
    externally. A future runner must additionally authenticate saved file bytes.
    All other expected bindings are mandatory and checked independently.
    """
    started = time.perf_counter()
    _unseal(payload)
    require(payload["integrity_sha256"] == _digest(expected_checkpoint_sha256), "Authenticated checkpoint")
    require(expected_kind == KIND and payload["kind"] == KIND and payload["version"] == VERSION
            and type(expected_settings) is TrainingSettings
            and payload["settings"] == expected_settings.configuration(), "Actual model kind/settings")
    require(payload["failed"] is False and payload["failure"] is None, "Failed partial trainer is terminal")
    require(payload["provenance"] == expected_provenance and payload["source_sha256"] == expected_source_sha256
            and payload["runtime"] == expected_runtime, "Expected provenance/source/runtime")
    initial = payload["initialization"]
    require(isinstance(initial, dict) and set(initial) == {"weights", "tensor_sha256"}
            and initial["tensor_sha256"] == _digest(expected_initial_sha256)
            and payload["data_sha256"] == _digest(expected_data_sha256), "Expected original initialization/data")
    with torch.random.fork_rng(devices=[]):
        result = TwoObservationTrainer(
            initial["weights"], payload["orders"], public_data, settings=expected_settings,
            expected_initial_sha256=expected_initial_sha256, expected_orders_sha256=expected_orders_sha256,
            expected_data_sha256=expected_data_sha256, provenance=expected_provenance,
            source_sha256=expected_source_sha256, runtime=expected_runtime,
        )
        require(set(payload) == set(result.export_checkpoint()), "Exact checkpoint membership")
        require(payload["model_configuration"] == result.model_configuration
                and payload["optimizer_group_names"] == result.optimizer_group_names, "Full model semantics/optimizer names")
        for key in ("successful_updates", "optimizer_steps"):
            require(type(payload[key]) is int and 0 <= payload[key] <= expected_settings.epochs * expected_settings.batches_per_epoch,
                    "Update count bounds")
            setattr(result, key, payload[key])
        require(result.successful_updates == result.optimizer_steps and payload["cursor"] == result.cursor,
                "Exact next order cursor")
        require(canonical_state_hash(payload["rng"]) == canonical_state_hash(result.rng_identity()), "Exact cursor-linked RNG identity")
        _load_weights(result.student, payload["student_state"])
        optimizer = payload["optimizer_state"]
        require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "Optimizer state schema")
        count = len(result.named_trainable)
        require(len(optimizer["param_groups"]) == 1 and optimizer["param_groups"][0]["params"] == list(range(count)),
                "Optimizer parameter ID order")
        require(set(optimizer["state"]) == (set(range(count)) if result.successful_updates else set()), "Optimizer state coverage")
        for index, parameter in enumerate(result.named_trainable.values()):
            if not result.successful_updates:
                break
            slots = optimizer["state"][index]
            require(set(slots) == {"step", "exp_avg", "exp_avg_sq"}, "Exact Adam slots")
            for key, shape in (("step", ()), ("exp_avg", parameter.shape), ("exp_avg_sq", parameter.shape)):
                value = slots[key]
                require(isinstance(value, torch.Tensor) and value.shape == shape and value.dtype == torch.float32
                        and value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Finite CPU Adam schema")
            require(float(slots["step"]) == result.successful_updates and bool((slots["exp_avg_sq"] >= 0).all()),
                    "Adam step/second moment")
        result.optimizer.load_state_dict(clone(optimizer))
        require(canonical_state_hash(result.optimizer.state_dict()) == canonical_state_hash(optimizer), "No optimizer coercion")
        result.log_chain_sha256 = _digest(payload["log_chain_sha256"])
        if not result.successful_updates:
            require(result.log_chain_sha256 == canonical_state_hash([]) and payload["last_attempt"] is None,
                    "Fresh trainer has no prior updates")
            require(canonical_tensor_hash(result.student.state_dict()) == expected_initial_sha256,
                    "Zero-update checkpoint must contain original initial tensors")
        else:
            attempt = payload["last_attempt"]
            require(isinstance(attempt, dict) and set(attempt) == {"status", "cursor", "completed_neural_sample_calls", "wall_seconds"}
                    and attempt["status"] == "completed", "Completed final attempt evidence")
            previous_epoch, previous_batch = divmod(result.successful_updates - 1, expected_settings.batches_per_epoch)
            require(attempt["cursor"] == {"epoch": previous_epoch, "batch": previous_batch}, "Final attempt cursor")
            batch = min(expected_settings.batch_size, expected_settings.train_episodes - previous_batch * expected_settings.batch_size)
            expected_work = result._work(batch)["operations"]
            require(attempt["completed_neural_sample_calls"] == {key: expected_work[key] for key in ("gru_cell_sample_calls", "linear_layer_sample_calls")},
                    "Final attempt neural work")
            _number(attempt["wall_seconds"], "attempt wall")
        for key in ("setup_wall_seconds", "training_wall_seconds", "restoration_wall_seconds"):
            _number(payload[key], key)
            setattr(result, key, payload[key])
        require(payload["last_attempt"] is None or payload["last_attempt"]["wall_seconds"] <= result.training_wall_seconds,
                "Attempt time exceeds complete recorded training time")
        result.last_attempt = clone(payload["last_attempt"])
        result._check_live()
        result.restoration_wall_seconds += time.perf_counter() - started
        return result
