"""Independent saved arithmetic for a completed two-observation training fit.

The caller authenticates file bytes, original initialization/data/order lineage
and all expected bindings. This module never loads files, imports a learned
model/trainer, constructs an optimizer or repeats training. Only the supplied
historical minibatch RNG is replayed in an isolated CPU Generator to verify
permutations and logical epoch states; this is not a fresh scientific stream.

Saved losses, gradient norms and Adam moments are checked for schema and
arithmetic consistency, not recomputed from a forward/backward/optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping

import torch

VERSION = "reacher-two-observation-training-audit-v1"
TRAINING_VERSION = "reacher-two-observation-training-v1"
MODEL_CLASS = "TwoObservationHistoryGRUWorldModel"
KIND = "two_observation_gru"
ADAM = {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
        "amsgrad": False, "foreach": False, "fused": False, "maximize": False,
        "capturable": False, "differentiable": False, "decoupled_weight_decay": False}
SETTING_FIELDS = {
    "hidden_size", "mlp_width", "dt", "noise_std", "residual_reward", "train_episodes", "steps",
    "epochs", "batch_size", "learning_rate", "gradient_clip", "rollout_horizon", "rollout_weight",
    "reward_scale", "kl_weight", "kl_balance", "free_nats", "optimizer", "dtype", "device",
    "objective", "run_status_authority", "adapter_version", "model_class", "mlp_width_usage",
}
STATE_KEYS = {"hidden", "packet", "pending_action", "imagined_depth", "real_packets", "real_actions",
              "real_present", "real_indices", "real_index", "real_target"}
CHECKPOINT_FIELDS = {
    "version", "kind", "settings", "model_configuration", "initialization", "orders", "data_sha256",
    "provenance", "source_sha256", "runtime", "student_state", "optimizer_state", "optimizer_group_names",
    "successful_updates", "optimizer_steps", "cursor", "rng", "failed", "failure", "last_attempt",
    "log_chain_sha256", "setup_wall_seconds", "training_wall_seconds", "restoration_wall_seconds",
}
LOG_FIELDS = {"kind", "epoch", "batch", "update", "optimizer_steps", "indices", "indices_sha256", "anchor",
              "loss", "gradient_norm", "gradient_clipped", "work", "costs", "previous_log_sha256",
              "log_sha256", "observed_neural_sample_calls", "rng_identity_sha256"}
COST_FIELDS = ("forward_seconds", "backward_seconds", "gradient_clip_seconds", "optimizer_seconds", "batch_wall_seconds")
KERNEL_FIELDS = ("gru_cell_sample_calls", "linear_layer_sample_calls")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    require(type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Lowercase SHA256")
    return value


def _number(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (not positive or value > 0), "Finite nonnegative " + label)
    return float(value)


def tensor_hash(values):
    """Independent implementation of the frozen named-tensor byte format."""
    require(isinstance(values, Mapping) and all(type(key) is str for key in values), "Named tensors")
    digest = hashlib.sha256(b"OpenJev named tensors v1\0")
    for name in sorted(values):
        value = values[name]
        require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                and value.device.type == "cpu", "Dense CPU tensor")
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "Finite tensor")
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], ensure_ascii=False, separators=(",", ":")).encode()
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(metadata)) + metadata + struct.pack(">Q", len(raw)) + raw)
    return digest.hexdigest()


def state_hash(value):
    """Independent type-sensitive nested checkpoint/log hashing."""
    def describe(item):
        if isinstance(item, torch.Tensor):
            return ["tensor", tensor_hash({"value": item})]
        if isinstance(item, dict):
            require(all(type(key) in (str, int) for key in item), "Safe checkpoint keys")
            return ["dict", [[type(key).__name__, key, describe(item[key])]
                             for key in sorted(item, key=lambda key: (type(key).__name__, str(key)))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [describe(part) for part in item]]
        require(item is None or type(item) in (bool, str, int, float), "Safe scalar metadata")
        require(type(item) is not float or math.isfinite(item), "Finite scalar metadata")
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(describe(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _same(left, right, label):
    require(state_hash(left) == state_hash(right), label)


def _unseal(payload):
    require(isinstance(payload, dict) and "integrity_sha256" in payload, "Sealed payload")
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    require(state_hash(body) == _digest(payload["integrity_sha256"]), "Payload integrity")
    return body


def validate_settings(settings):
    require(type(settings) is dict and set(settings) == SETTING_FIELDS, "Exact external training settings")
    for key in ("hidden_size", "mlp_width", "train_episodes", "steps", "epochs", "batch_size", "rollout_horizon"):
        require(type(settings[key]) is int and settings[key] > 0, "Positive integer " + key)
    require(settings["steps"] <= 50, "At most fifty training actions")
    for key in ("dt", "learning_rate", "gradient_clip"):
        _number(settings[key], key, positive=True)
    for key in ("noise_std", "rollout_weight", "reward_scale", "kl_weight", "kl_balance", "free_nats"):
        _number(settings[key], key)
    require(settings["kl_balance"] <= 1 and settings["residual_reward"] is True, "Balance/residual settings")
    expected = {"optimizer": ADAM, "dtype": "torch.float32", "device": "cpu", "objective": "unchanged_sequence_loss",
        "run_status_authority": "enclosing protocol and execution receipts", "adapter_version": TRAINING_VERSION,
        "model_class": MODEL_CLASS, "mlp_width_usage": "unused; retained original paired settings metadata"}
    _same({key: settings[key] for key in expected}, expected, "Exact optimizer/actual-class settings")


def parameter_shapes(hidden):
    """Ordered named-parameter schema, derived algebraically without a model."""
    result = {}
    for name, inputs in (("observation_update", 8), ("transition", 6)):
        result[name + ".weight_ih"] = (3 * hidden, inputs)
        result[name + ".weight_hh"] = (3 * hidden, hidden)
        result[name + ".bias_ih"] = result[name + ".bias_hh"] = (3 * hidden,)
    for name, inputs, outputs in (("observation_head.0", hidden, hidden), ("observation_head.2", hidden, 4),
                                  ("reward_head.0", hidden + 2, hidden), ("reward_head.2", hidden, 1)):
        result[name + ".weight"], result[name + ".bias"] = (outputs, inputs), (outputs,)
    return result


def model_configuration(settings):
    return {"version": "two-valid-observation-history-gru-v1", "model_class": MODEL_CLASS,
        "hidden_size": settings["hidden_size"], "dt": float(settings["dt"]), "noise_std": float(settings["noise_std"]), "residual_reward": True,
        "valid_observations": 2, "max_real_packets": 12, "max_issued_commands": 11,
        "episode_steps": 50, "age_rtol": 1e-5, "age_atol": 1e-6,
        "anchor": "older of last two actual valid observations; first observation before second exists",
        "real_history": "sanitized actual public packets and issued commands; integer indices; left padding",
        "assimilation": "zero-start reconstruction at every real boundary; initial-or-one-selected-advance phase",
        "replay": "all twelve assimilations and eleven full transitions, including padding, without detach",
        "imagined_rollout": "private recurrent hidden/packet; never appends real evidence",
        "overflow": "reject before dropping required anchor or command", "state_keys": sorted(STATE_KEYS),
        "run_status_authority": "enclosing protocol and execution receipts"}


def _tensor(value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided and value.device.type == "cpu"
            and tuple(value.shape) == tuple(shape) and value.dtype == dtype and bool(torch.isfinite(value).all()), label)


def _weights(values, schema, label):
    require(isinstance(values, Mapping) and set(values) == set(schema), "Exact " + label + " tensor membership")
    for name, shape in schema.items():
        _tensor(values[name], shape, torch.float32, label + " " + name)


def _public_data(data, settings):
    require(type(data) is dict and set(data) == {"packets", "commands", "rewards"}, "Public data allowlist")
    n, t = settings["train_episodes"], settings["steps"]
    for key, shape in {"packets": (n, t + 1, 8), "commands": (n, t, 2), "rewards": (n, t)}.items():
        _tensor(data[key], shape, torch.float32, "Complete public " + key)
    packet = data["packets"]
    valid = packet[..., 6] == 1
    require(bool(((packet[..., 6] == 0) | valid).all()) and bool(valid[:, 0].all())
            and bool((packet[..., 7] >= 0).all()), "Binary validity, nonnegative age and initial observation")
    require(bool((packet[..., :4][~valid] == 0).all()) and bool((data["commands"].abs() <= 1).all()), "Masked angles/clipped issued actions")
    require(torch.equal(packet[..., 4:6], packet[:, :1, 4:6].expand(-1, t + 1, -1)), "Static public target")
    latest = torch.full((n,), -1, dtype=torch.int64)
    older = latest.clone()
    for step in range(t + 1):
        older = torch.where(valid[:, step], latest, older)
        latest = torch.where(valid[:, step], step, latest)
        age = (step - latest).float() * settings["dt"]
        require(bool(torch.isclose(packet[:, step, 7], age, rtol=1e-5, atol=1e-6).all())
                and bool((packet[:, step, 7][valid[:, step]] == 0).all()), "Actual public observation age")
        anchor = torch.where(older >= 0, older, latest)
        require(bool((step - anchor < 12).all()), "Two-observation span overflow")


def replay_orders(orders, settings, *, deadline_check=None):
    """Verify only the bound historical permutations with an isolated RNG."""
    check = deadline_check or (lambda: None)
    body = _unseal(orders)
    require(set(body) == {"version", "seed", "role", "orders", "initial_rng", "final_rng"}, "Exact order schema")
    require(type(body["seed"]) is int and 0 <= body["seed"] < 2**32, "Historical uint32 seed")
    require(all(type(body[key]) is str and body[key] for key in ("version", "role")), "Historical order identities")
    n, epochs = settings["train_episodes"], settings["epochs"]
    _tensor(body["orders"], (epochs, n), torch.int64, "Complete saved epoch orders")
    generator = torch.Generator(device="cpu").manual_seed(body["seed"])
    states = [generator.get_state().clone()]
    for epoch in range(epochs):
        check()
        wanted = torch.randperm(n, generator=generator)
        require(torch.equal(body["orders"][epoch], wanted), "Exact historical epoch permutation")
        states.append(generator.get_state().clone())
    for key, wanted in (("initial_rng", states[0]), ("final_rng", states[-1])):
        _tensor(body[key], wanted.shape, torch.uint8, "Saved order RNG " + key)
        require(torch.equal(body[key], wanted), "Historical order RNG endpoint")
    return states


def rng_identity(orders, states, update, batches):
    consumed = (update + batches - 1) // batches
    return {"neural_rng": "no draws; deterministic CPU model; ambient RNG preserved",
        "order_role": orders["role"], "order_seed": orders["seed"], "consumed_epoch_permutations": consumed,
        "initial_rng": states[0], "current_rng": states[consumed], "final_rng": states[-1],
        "convention": "immutable full orders; logical RNG advances once upon first batch of each epoch"}


def expected_work(settings, batch):
    t, h, width = settings["steps"], settings["rollout_horizon"], settings["hidden_size"]
    starts = max(0, t - h + 1)
    roll = h if starts and settings["rollout_weight"] else 0
    calls = {"student_assimilate": t, "student_prefix_advance": t, "student_rollout_advance": roll, "student_posterior_kl": t}
    samples = {key: count * batch for key, count in calls.items()}
    samples["student_rollout_advance"] *= starts
    a = samples["student_assimilate"]
    d = samples["student_prefix_advance"] + samples["student_rollout_advance"]
    updates, replayed, transitions = 12 * a, 11 * a, 11 * a + d
    named = {key: math.prod(shape) for key, shape in parameter_shapes(width).items()}
    operations = {"configuration": model_configuration(settings), "trainable_parameters": sum(named.values()), "named_parameters": named,
        "public_assimilate_samples": a, "advance_samples": d, "replayed_observation_update_samples": updates,
        "replayed_transition_samples": replayed, "gru_cell_sample_calls": updates + transitions,
        "linear_layer_sample_calls": 4 * transitions, "analytic_reward_sample_calls": transitions,
        "dense_affine_macs": updates * 3 * width * (width + 8) + transitions * (5 * width * width + 25 * width),
        "startup_masked_work_is_counted": True, "compute_matched": False,
        "counts_are_not_total_flops_or_measured_wall_time": True}
    return {"interfaces": {"batch_forward_calls": calls, "sample_forward_evaluations": samples, "counts_are_not_flops": True},
            "operations": operations, "compute_matched": False}


def audit_training(checkpoint, logs, *, initial_weights, orders, public_data, settings,
                   expected_initial_sha256, expected_orders_sha256, expected_data_sha256,
                   expected_checkpoint_sha256, provenance, source_sha256, runtime,
                   final_weights=None, deadline_check=None):
    """Verify a complete successful saved fit, not a partial/resume authority.

    ``expected_checkpoint_sha256`` is the sealed BODY's integrity digest.
    ``expected_orders_sha256`` hashes the WHOLE sealed order payload. File-byte
    authentication remains the caller's responsibility. ``settings`` is the
    exact external TrainingSettings.configuration() mapping, not a class.
    """
    check = deadline_check or (lambda: None)
    check()
    validate_settings(settings)
    for value in (expected_initial_sha256, expected_orders_sha256, expected_data_sha256, expected_checkpoint_sha256):
        _digest(value)
    for label, value in (("provenance", provenance), ("sources", source_sha256), ("runtime", runtime)):
        require(type(value) is dict and value and all(type(key) is str and key for key in value), "Explicit external " + label)
        state_hash(value)
    for digest in source_sha256.values():
        _digest(digest)
    schema = parameter_shapes(settings["hidden_size"])
    _weights(initial_weights, schema, "original initialization")
    require(tensor_hash(initial_weights) == expected_initial_sha256, "Externally pinned original tensors")
    require(state_hash(orders) == expected_orders_sha256, "Externally pinned whole sealed orders")
    states = replay_orders(orders, settings, deadline_check=check)
    _public_data(public_data, settings)
    require(tensor_hash(public_data) == expected_data_sha256, "Externally pinned public data")
    body = _unseal(checkpoint)
    require(checkpoint["integrity_sha256"] == expected_checkpoint_sha256 and set(body) == CHECKPOINT_FIELDS, "Exact authenticated checkpoint")
    require(body["version"] == TRAINING_VERSION and body["kind"] == KIND
            and body["failed"] is False and body["failure"] is None, "Successful actual-class checkpoint")
    for key, wanted in (("settings", settings), ("model_configuration", model_configuration(settings)),
                        ("provenance", provenance), ("source_sha256", source_sha256), ("runtime", runtime),
                        ("orders", orders), ("initialization", {"weights": initial_weights, "tensor_sha256": expected_initial_sha256})):
        _same(body[key], wanted, "Checkpoint external binding: " + key)
    require(body["data_sha256"] == expected_data_sha256, "Checkpoint public data binding")
    batches = math.ceil(settings["train_episodes"] / settings["batch_size"])
    total = settings["epochs"] * batches
    for key in ("successful_updates", "optimizer_steps"):
        require(type(body[key]) is int and body[key] == total, "Complete " + key)
    _same(body["cursor"], {"epoch": settings["epochs"], "batch": 0}, "Complete next minibatch cursor")
    _same(body["rng"], rng_identity(orders, states, total, batches), "Final cursor-linked RNG identity")
    _weights(body["student_state"], schema, "final student")
    final_digest = tensor_hash(body["student_state"])
    if final_weights is not None:
        _weights(final_weights, schema, "deployment weights")
        require(tensor_hash(final_weights) == final_digest, "Checkpoint/deployment weight equality")
    _same(body["optimizer_group_names"], [list(schema)], "Actual named parameter ordering")
    optimizer = body["optimizer_state"]
    require(type(optimizer) is dict and set(optimizer) == {"state", "param_groups"}, "Exact saved Adam structure")
    group = {key: value for key, value in ADAM.items() if key != "name"}
    group.update(lr=settings["learning_rate"], betas=tuple(ADAM["betas"]), params=list(range(len(schema))))
    _same(optimizer["param_groups"], [group], "Exact saved Adam recipe and parameter IDs")
    require(type(optimizer["state"]) is dict and all(type(key) is int for key in optimizer["state"])
            and set(optimizer["state"]) == set(range(len(schema))), "All saved Adam parameter slots")
    for index, shape in enumerate(schema.values()):
        slot = optimizer["state"][index]
        require(type(slot) is dict and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Exact Adam moment fields")
        for key, wanted_shape in (("step", ()), ("exp_avg", shape), ("exp_avg_sq", shape)):
            _tensor(slot[key], wanted_shape, torch.float32, "Adam " + key)
        require(float(slot["step"]) == total and bool((slot["exp_avg_sq"] >= 0).all()), "Adam steps/nonnegative second moment")
    require(isinstance(logs, (list, tuple)) and len(logs) == total, "Every successful update logged")
    previous = state_hash([])
    costs = dict.fromkeys(COST_FIELDS, 0.)
    work_totals = dict.fromkeys(("public_assimilate_samples", "advance_samples", "replayed_observation_update_samples",
        "replayed_transition_samples", "gru_cell_sample_calls", "linear_layer_sample_calls", "analytic_reward_sample_calls", "dense_affine_macs"), 0)
    for index, row in enumerate(logs):
        check()
        require(type(row) is dict and set(row) == LOG_FIELDS, "Exact complete update-log schema")
        epoch, batch = divmod(index, batches)
        for key, wanted in (("epoch", epoch), ("batch", batch), ("update", index + 1), ("optimizer_steps", index + 1)):
            require(type(row[key]) is int and row[key] == wanted, "Update-log cursor: " + key)
        indices = orders["orders"][epoch, batch * settings["batch_size"]:(batch + 1) * settings["batch_size"]]
        require(row["kind"] == KIND and row["indices_sha256"] == tensor_hash({"indices": indices}), "Actual paired index hash")
        _same(row["indices"], indices.tolist(), "Exact paired minibatch indices")
        require(row["previous_log_sha256"] == previous and row["log_sha256"] == state_hash({k: v for k, v in row.items() if k != "log_sha256"}), "Complete update log chain")
        previous = row["log_sha256"]
        require(row["rng_identity_sha256"] == state_hash(rng_identity(orders, states, index + 1, batches)), "Per-update logical epoch RNG")
        metrics = row["anchor"]
        require(type(metrics) is dict and set(metrics) == {"loss", "observation_mse", "reward_mse", "kl_nats", "rollout_observation_mse",
            "rollout_reward_mse", "valid_observation_targets", "valid_rollout_starts"}, "Exact anchor metrics")
        for key, value in metrics.items():
            _number(value, "anchor " + key)
        packet = public_data["packets"][indices]
        starts = max(0, settings["steps"] - settings["rollout_horizon"] + 1)
        require(metrics["valid_observation_targets"] == float(packet[:, 1:, 6].sum())
                and metrics["valid_rollout_starts"] == float(packet[:, :starts, 6].sum()), "Actual public training target masks")
        require(metrics["kl_nats"] == 0, "Deterministic GRU has zero KL")
        if not settings["rollout_weight"] or not starts:
            require(metrics["rollout_observation_mse"] == metrics["rollout_reward_mse"] == 0, "Disabled rollout losses")
        wanted_loss = metrics["observation_mse"] + settings["reward_scale"] * metrics["reward_mse"] + settings["kl_weight"] * metrics["kl_nats"]
        wanted_loss += settings["rollout_weight"] * (metrics["rollout_observation_mse"] + settings["reward_scale"] * metrics["rollout_reward_mse"])
        loss = _number(row["loss"], "logged loss")
        require(loss == metrics["loss"] and math.isclose(loss, wanted_loss, rel_tol=3e-6, abs_tol=1e-6), "Saved anchor loss arithmetic")
        norm = _number(row["gradient_norm"], "gradient norm")
        # The trainer compares a float32 scalar tensor with the Python scalar;
        # PyTorch rounds the latter to that tensor dtype before comparison.
        clipped = bool(torch.tensor(norm, dtype=torch.float32) > settings["gradient_clip"])
        require(type(row["gradient_clipped"]) is bool and row["gradient_clipped"] == clipped, "Gradient clip flag arithmetic")
        work = expected_work(settings, len(indices))
        _same(row["work"], work, "Full padded 12/11 work")
        kernels = {key: work["operations"][key] for key in KERNEL_FIELDS}
        _same(row["observed_neural_sample_calls"], kernels, "Observed completed neural kernels")
        for key in work_totals:
            work_totals[key] += work["operations"][key]
        require(type(row["costs"]) is dict and set(row["costs"]) == set(COST_FIELDS), "All update timing phases")
        for key in COST_FIELDS:
            costs[key] += _number(row["costs"][key], key, positive=True)
        require(sum(row["costs"][key] for key in COST_FIELDS[:-1]) <= row["costs"]["batch_wall_seconds"] + 1e-6, "Nested update timing")
    require(previous == body["log_chain_sha256"], "Final saved update chain")
    last = body["last_attempt"]
    require(type(last) is dict and set(last) == {"status", "cursor", "completed_neural_sample_calls", "wall_seconds"}
            and last["status"] == "completed", "Completed final attempt evidence")
    _same(last["cursor"], {"epoch": logs[-1]["epoch"], "batch": logs[-1]["batch"]}, "Final attempt cursor")
    _same(last["completed_neural_sample_calls"], logs[-1]["observed_neural_sample_calls"], "Final attempt kernel witnesses")
    require(_number(last["wall_seconds"], "final attempt wall", positive=True) == logs[-1]["costs"]["batch_wall_seconds"], "Final attempt/log timing")
    for key in ("setup_wall_seconds", "training_wall_seconds", "restoration_wall_seconds"):
        _number(body[key], key)
    require(costs["batch_wall_seconds"] <= body["training_wall_seconds"] + 1e-6, "All logged update time charged")
    check()
    return {"version": VERSION, "status": "saved_arithmetic_verified", "kind": KIND, "model_class": MODEL_CLASS,
        "updates": total, "optimizer_steps": total, "logged_updates": len(logs), "parameters": sum(math.prod(v) for v in schema.values()),
        "initial_tensor_sha256": expected_initial_sha256, "orders_payload_sha256": expected_orders_sha256,
        "data_sha256": expected_data_sha256, "checkpoint_integrity_sha256": expected_checkpoint_sha256,
        "final_tensor_sha256": final_digest, "deployment_weights_compared": final_weights is not None, "log_chain_sha256": previous,
        "historical_order_permutations_checked": settings["epochs"], "historical_rng_states_checked": len(states),
        "work": work_totals, "phase_seconds": costs, "setup_wall_seconds": body["setup_wall_seconds"],
        "training_wall_seconds": body["training_wall_seconds"], "restoration_wall_seconds": body["restoration_wall_seconds"],
        "new_model_calls": 0, "new_optimizer_steps": 0, "new_native_calls": 0, "new_scientific_streams": 0,
        "limits": ["Caller authenticates saved file bytes and original data/initialization/source/runtime provenance.",
            "Saved tensor/schema/log arithmetic only; no forward/backward or numerical optimizer-transition replay.",
            "Historical isolated minibatch RNG replay verifies supplied orders, not a new initialization or evaluation stream.",
            "Completed checkpoint does not certify native control, scientific qualification or launch authorization."]}
