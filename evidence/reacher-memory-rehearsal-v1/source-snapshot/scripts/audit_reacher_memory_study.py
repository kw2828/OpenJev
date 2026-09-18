"""Independent saved-output audit of the prospective Reacher memory comparison.

No training module, runner or learned model is imported or instantiated. Native
replay consumes saved commands. CEM reconstruction consumes recorded scalar
scores. Initialization is reconstructed as tensors from named generators, not
as a neural module. Optimizer transitions and predictions remain source-bound
saved evidence, rather than independently repeated training or inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextvars import ContextVar
from pathlib import Path

import audit_reacher_objective_study as previous
import numpy as np
import torch

from openjev.research import reacher_memory_protocol as protocol
from openjev.research.reacher_random_streams import concrete_streams

base, search_audit = previous.base, previous.search_audit
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-memory-ablation-v1"
TRAINING_VERSION = "reacher-memory-training-engineering-v1"
CONTROL_VERSION = "reacher-memory-control-engineering-v1"
TRAINING_FIELDS = (
    "hidden_size",
    "mlp_width",
    "dt",
    "noise_std",
    "residual_reward",
    "train_episodes",
    "steps",
    "epochs",
    "batch_size",
    "learning_rate",
    "gradient_clip",
    "rollout_horizon",
    "rollout_weight",
    "reward_scale",
    "kl_weight",
    "kl_balance",
    "free_nats",
)
SEMANTICS = {
    "residual_gru": "persistent real-prefix GRU; missing observations preserve predicted recurrent state",
    "current_gru": "zero hidden before every real packet, including missing; recurrent imagined advances",
    "bounded_gru": "rebuild from final3 real packets and2 issued commands; explicit startup and real phases",
    "packet_mlp": "overwrite state with every real packet; action-conditioned imagined packets only",
}
NEW_SOURCES = {
    f"{folder}/{name}.py"
    for folder, names in (
        (
            "src/openjev/research",
            (
                "reacher_observation_baseline",
                "reacher_bounded_history",
                "reacher_memory_training",
                "reacher_memory_control",
                "reacher_memory_protocol",
                "reacher_kinematic_control",
            ),
        ),
        ("scripts", ("reacher_memory_study", "audit_reacher_memory_study")),
        (
            "tests",
            (
                "test_reacher_observation_baseline",
                "test_reacher_bounded_history",
                "test_reacher_memory_training",
                "test_reacher_memory_control",
                "test_reacher_memory_protocol",
                "test_reacher_kinematic_control",
                "test_reacher_memory_study",
                "test_audit_reacher_memory_study",
            ),
        ),
    )
    for name in names
}
_DEADLINE = ContextVar("memory_audit_deadline", default=float("inf"))
require, read, write, sha = base.require, base.read, base.write, base.sha
canonical_tensor_hash, canonical_state_hash = previous.canonical_tensor_hash, previous.canonical_state_hash
finite, exact_integer, tensors = previous.finite, previous.exact_integer, previous.tensor_mapping
load_tensors = previous.read_tensor_file
array = search_audit.array


def check_budget():
    if time.monotonic() > _DEADLINE.get():
        raise TimeoutError("Frozen memory audit cap exceeded")


def checked(path, digest):
    check_budget()
    return search_audit.checked(Path(path), digest)


def unseal(payload):
    require(isinstance(payload, dict) and "integrity_sha256" in payload, "Sealed checkpoint mapping")
    value = {k: v for k, v in payload.items() if k != "integrity_sha256"}
    require(canonical_state_hash(value) == payload["integrity_sha256"], "Checkpoint integrity")
    return value


def training_configuration(plan):
    return {
        **{key: plan[key] for key in TRAINING_FIELDS},
        "optimizer": plan["optimizer"],
        "dtype": "torch.float32",
        "device": "cpu",
        "objective": "unchanged_sequence_loss",
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def model_configuration(plan, arm):
    return {
        "kind": arm,
        "model_class": protocol.MODEL_CLASSES[arm],
        "width": plan["mlp_width"] if arm == "packet_mlp" else plan["hidden_size"],
        "dt": plan["dt"],
        "noise_std": plan["noise_std"],
        "residual_reward": True,
        "real_assimilation": SEMANTICS[arm],
        "imagined_rollout": "advance only; no imaginary observation is assimilated",
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def shapes(plan, arm):
    if arm != "packet_mlp":
        return previous.student_shapes(plan)
    width = plan["mlp_width"]
    result = {}
    for name, inputs, outputs in (
        ("encoder.0", 10, width),
        ("encoder.2", width, width),
        ("observation_head.0", width, width),
        ("observation_head.2", width, 4),
        ("reward_head.0", width + 2, width),
        ("reward_head.2", width, 1),
    ):
        result[name + ".weight"], result[name + ".bias"] = (outputs, inputs), (outputs,)
    return result


def seeded_tensors(plan, arm, seed):
    """Reproduce PyTorch GRUCell/Linear constructor draws without a model."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    schema = shapes(plan, arm)
    for name, shape in schema.items():
        value = torch.empty(shape, dtype=torch.float32)
        if name.startswith(("observation_update.", "transition.")):
            bound = 1 / math.sqrt(plan["hidden_size"])
            value.uniform_(-bound, bound, generator=generator)
        elif name.endswith(".weight"):
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            fan_in = schema[name.removesuffix(".bias") + ".weight"][1]
            value.uniform_(-1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), generator=generator)
        result[name] = value
    return result


def audit_preparation(plan, execution):
    initials, orders = {}, {}
    for pair in protocol.PAIRS:
        check_budget()
        initial = load_tensors(execution / "initializations" / f"{pair}.pt")
        body = unseal(initial)
        require(
            set(body) == {"version", "settings", "roles", "seeds", "states", "tensor_hashes", "wall_seconds"}
            and body["version"] == TRAINING_VERSION
            and body["settings"] == training_configuration(plan),
            "Complete paired initialization schema",
        )
        require(
            body["roles"] == {family: f"fit/{family}/{pair}" for family in ("gru", "mlp")}
            and body["seeds"]
            == {family: protocol.seed(plan, f"fit/{family}/{pair}") for family in ("gru", "mlp")}
            and set(body["states"]) == set(body["tensor_hashes"]) == {"gru", "mlp"},
            "Named initialization roles",
        )
        for family, arm in (("gru", "residual_gru"), ("mlp", "packet_mlp")):
            tensors(body["states"][family], shapes(plan, arm), "Initial parameter schema")
            require(
                canonical_tensor_hash(body["states"][family])
                == body["tensor_hashes"][family]
                == canonical_tensor_hash(seeded_tensors(plan, arm, body["seeds"][family])),
                "Exact named-seed initial parameter values",
            )
        finite(body["wall_seconds"], "Initialization time", positive=True)
        order = load_tensors(execution / "orders" / f"{pair}.pt")
        unseal(order)
        require(
            set(order)
            == {"version", "seed", "role", "orders", "initial_rng", "final_rng", "integrity_sha256"}
            and order["version"] == TRAINING_VERSION
            and order["role"] == f"fit/minibatch/{pair}"
            and order["seed"] == protocol.seed(plan, order["role"]),
            "Named order payload",
        )
        generator = torch.Generator().manual_seed(order["seed"])
        require(torch.equal(order["initial_rng"], generator.get_state()), "Initial minibatch generator")
        wanted = torch.stack(
            [torch.randperm(plan["train_episodes"], generator=generator) for _ in range(plan["epochs"])]
        )
        require(
            order["orders"].dtype == torch.int64
            and torch.equal(order["orders"], wanted)
            and torch.equal(order["final_rng"], generator.get_state()),
            "Exact complete minibatch orders",
        )
        for folder, payload in (("initializations", initial), ("orders", order)):
            path = execution / folder / f"{pair}.pt"
            require(
                read(path.with_suffix(".json"))
                == {"pair": pair, "file_sha256": sha(path), "integrity_sha256": payload["integrity_sha256"]},
                "Preparation sidecar identity",
            )
        initials[pair], orders[pair] = initial, order
    return initials, orders


def audit_checkpoint(plan, payload, initial, order, arm, data_hash):
    body = unseal(payload)
    expected = {
        "version",
        "kind",
        "settings",
        "model_configuration",
        "initialization",
        "orders",
        "data_sha256",
        "source_sha256",
        "runtime",
        "student_state",
        "optimizer_state",
        "optimizer_group_names",
        "successful_updates",
        "optimizer_steps",
        "cursor",
        "failed",
        "log_chain_sha256",
        "setup_wall_seconds",
        "training_wall_seconds",
        "restoration_wall_seconds",
    }
    require(
        set(body) == expected
        and body["version"] == TRAINING_VERSION
        and body["kind"] == arm
        and body["settings"] == training_configuration(plan)
        and body["model_configuration"] == model_configuration(plan, arm),
        "Actual class and complete checkpoint configuration",
    )
    require(
        canonical_state_hash(body["initialization"]) == canonical_state_hash(initial)
        and canonical_state_hash(body["orders"]) == canonical_state_hash(order)
        and body["data_sha256"] == data_hash
        and body["source_sha256"] == plan["sources"]
        and body["runtime"] == plan["runtime"]
        and body["failed"] is False,
        "Checkpoint external bindings",
    )
    updates = protocol.coverage(plan)["optimizer_updates_per_fit"]
    for key in ("successful_updates", "optimizer_steps"):
        exact_integer(body[key], updates, "Complete optimizer schedule")
    require(body["cursor"] == {"epoch": plan["epochs"], "batch": 0}, "Next minibatch cursor at completion")
    schema = shapes(plan, arm)
    tensors(body["student_state"], schema, "Final student")
    require(body["optimizer_group_names"] == [list(schema)], "Actual parameter ownership/order")
    optimizer = body["optimizer_state"]
    require(
        set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
        "Adam group schema",
    )
    group = {key: value for key, value in plan["optimizer"].items() if key != "name"}
    group.update(lr=plan["learning_rate"], betas=tuple(group["betas"]), params=list(range(len(schema))))
    require(
        optimizer["param_groups"][0] == group and set(optimizer["state"]) == set(group["params"]),
        "Exact Adam recipe/state coverage",
    )
    for index, (name, shape) in enumerate(schema.items()):
        slot = optimizer["state"][index]
        tensors(slot, {"step": (), "exp_avg": shape, "exp_avg_sq": shape}, "Adam slot " + name)
        require(
            float(slot["step"]) == updates and bool((slot["exp_avg_sq"] >= 0).all()),
            "Adam step/second moment",
        )
    for key in ("setup_wall_seconds", "training_wall_seconds", "restoration_wall_seconds"):
        finite(body[key], key)
    require(body["restoration_wall_seconds"] == 0, "Saved fit must precede restoration")
    return body


def bounded_configuration(plan):
    return {
        "version": "three-real-packet-gru-v1",
        "model_class": "BoundedThreePacketGRUWorldModel",
        "hidden_size": plan["hidden_size"],
        "dt": plan["dt"],
        "noise_std": plan["noise_std"],
        "residual_reward": True,
        "real_packet_window": 3,
        "intervening_issued_commands": 2,
        "real_history": "sanitized public packets only; left-padded explicit startup masks",
        "assimilation": "rebuild from zero at every real packet; initial-or-one-advance phase required",
        "imagined_rollout": "private recurrent hidden/packet state; never append imagined packets to history",
        "state_keys": sorted(
            (
                "hidden",
                "packet",
                "real_packets",
                "real_actions",
                "real_valid",
                "pending_action",
                "imagined_depth",
            )
        ),
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def operations(plan, arm, a, d):
    h, w = plan["hidden_size"], plan["mlp_width"]
    if arm == "bounded_gru":
        named = {name: math.prod(shape) for name, shape in shapes(plan, arm).items()}
        return {
            "configuration": bounded_configuration(plan),
            "trainable_parameters": sum(named.values()),
            "named_parameters": named,
            "public_assimilate_samples": a,
            "advance_samples": d,
            "replayed_observation_update_samples": 3 * a,
            "replayed_transition_samples": 2 * a,
            "gru_cell_sample_calls": 5 * a + d,
            "linear_layer_sample_calls": 4 * (2 * a + d),
            "analytic_reward_sample_calls": 2 * a + d,
            "dense_affine_macs": 3 * a * 3 * h * (h + 8) + (2 * a + d) * (5 * h * h + 25 * h),
            "startup_masked_work_is_counted": True,
            "compute_matched": False,
            "counts_are_not_total_flops_or_measured_wall_time": True,
        }
    result = {
        "gru_cell_sample_calls": 0 if arm == "packet_mlp" else a + d,
        "linear_layer_sample_calls": (6 if arm == "packet_mlp" else 4) * d,
        "analytic_reward_sample_calls": d,
        "dense_affine_macs": d * (3 * w * w + 17 * w)
        if arm == "packet_mlp"
        else a * 3 * h * (h + 8) + d * (5 * h * h + 25 * h),
        "counts_are_not_total_flops_or_measured_wall_time": True,
    }
    if arm != "residual_gru":
        result.update(assimilate_samples=a, advance_samples=d, real_state_reset_samples=a)
    return result


def audit_work(plan, arm, value, a, d):
    require(
        value["assimilate_samples"] == a
        and value["advance_samples"] == d
        and value["operations"] == operations(plan, arm, a, d)
        and value["compute_matched"] is False,
        "Actual-class operation/sample accounting",
    )
    require(
        set(value) == {"assimilate_samples", "advance_samples", "operations", "compute_matched", "limits"}
        and isinstance(value["limits"], str),
        "Model work schema",
    )


def anchor_interfaces(plan, batch):
    t, h = plan["steps"], plan["rollout_horizon"]
    starts = max(0, t - h + 1)
    roll = h if starts and plan["rollout_weight"] else 0
    calls = {
        "student_assimilate": t,
        "student_prefix_advance": t,
        "student_rollout_advance": roll,
        "student_posterior_kl": t,
    }
    samples = {name: count * batch for name, count in calls.items()}
    samples["student_rollout_advance"] *= starts
    return {"batch_forward_calls": calls, "sample_forward_evaluations": samples, "counts_are_not_flops": True}


def audit_logs(plan, path, arm, order, checkpoint, packets=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    batches = plan["train_episodes"] // plan["batch_size"]
    require(len(rows) == plan["epochs"] * batches, "Every training update logged")
    previous_hash = canonical_state_hash([])
    costs = {
        key: 0.0
        for key in (
            "forward_seconds",
            "backward_seconds",
            "gradient_clip_seconds",
            "optimizer_seconds",
            "batch_wall_seconds",
        )
    }
    for offset, row in enumerate(rows):
        check_budget()
        require(
            set(row)
            == {
                "kind",
                "epoch",
                "batch",
                "update",
                "optimizer_steps",
                "indices",
                "indices_sha256",
                "anchor",
                "loss",
                "gradient_norm",
                "gradient_clipped",
                "work",
                "costs",
                "previous_log_sha256",
                "log_sha256",
            },
            "Training log schema",
        )
        epoch, batch = divmod(offset, batches)
        indices = order["orders"][epoch, batch * plan["batch_size"] : (batch + 1) * plan["batch_size"]]
        require(
            row["kind"] == arm
            and row["epoch"] == epoch
            and row["batch"] == batch
            and row["update"] == row["optimizer_steps"] == offset + 1
            and row["indices"] == indices.tolist()
            and row["indices_sha256"] == canonical_tensor_hash({"indices": indices}),
            "Paired data and update alignment",
        )
        require(
            row["previous_log_sha256"] == previous_hash
            and row["log_sha256"]
            == canonical_state_hash({key: value for key, value in row.items() if key != "log_sha256"}),
            "Training log chain",
        )
        previous_hash = row["log_sha256"]
        metrics = row["anchor"]
        require(
            set(metrics)
            == {
                "loss",
                "observation_mse",
                "reward_mse",
                "kl_nats",
                "rollout_observation_mse",
                "rollout_reward_mse",
                "valid_observation_targets",
                "valid_rollout_starts",
            },
            "Anchor metrics",
        )
        for key, value in metrics.items():
            finite(value, "Anchor " + key)
        if packets is not None:
            selected = packets[indices.numpy()]
            starts = max(0, plan["steps"] - plan["rollout_horizon"] + 1)
            require(
                metrics["valid_observation_targets"] == float(selected[:, 1:, 6].sum())
                and metrics["valid_rollout_starts"] == float(selected[:, :starts, 6].sum()),
                "Public training target masks",
            )
        wanted_loss = (
            metrics["observation_mse"]
            + plan["reward_scale"] * metrics["reward_mse"]
            + plan["kl_weight"] * metrics["kl_nats"]
        )
        wanted_loss += plan["rollout_weight"] * (
            metrics["rollout_observation_mse"] + plan["reward_scale"] * metrics["rollout_reward_mse"]
        )
        require(
            math.isclose(row["loss"], wanted_loss, rel_tol=3e-6, abs_tol=1e-6)
            and row["loss"] == metrics["loss"]
            and metrics["kl_nats"] == 0,
            "Anchor loss arithmetic",
        )
        norm = finite(row["gradient_norm"], "Gradient norm")
        require(
            type(row["gradient_clipped"]) is bool
            and row["gradient_clipped"] == (norm > plan["gradient_clip"]),
            "Clip flag",
        )
        interfaces = anchor_interfaces(plan, len(indices))
        samples = interfaces["sample_forward_evaluations"]
        ops = operations(
            plan,
            arm,
            samples["student_assimilate"],
            samples["student_prefix_advance"] + samples["student_rollout_advance"],
        )
        require(
            row["work"] == {"interfaces": interfaces, "operations": ops, "compute_matched": False},
            "Training operation counts",
        )
        require(set(row["costs"]) == set(costs), "Training phase cost membership")
        for key in costs:
            costs[key] += finite(row["costs"][key], key, positive=True)
        require(
            sum(row["costs"][key] for key in costs if key != "batch_wall_seconds")
            <= row["costs"]["batch_wall_seconds"] + 1e-6,
            "Nested update cost",
        )
    require(
        previous_hash == checkpoint["log_chain_sha256"]
        and costs["batch_wall_seconds"] <= checkpoint["training_wall_seconds"] + 1e-6,
        "Final chain and training cost",
    )
    return {
        "updates": len(rows),
        "phase_seconds": costs,
        "final_loss": rows[-1]["loss"],
        "log_chain_sha256": previous_hash,
    }


def audit_fit(plan, execution, row, initial, order, data_hash, packets=None):
    folder = execution / "fits" / row["name"]
    done = read(folder / "completed.json")
    require(
        set(done)
        == {
            "name",
            "arm",
            "pair",
            "model_configuration",
            "settings",
            "updates",
            "optimizer_steps",
            "parameters",
            "initialization_sha256",
            "orders_sha256",
            "data_sha256",
            "student_tensor_sha256",
            "checkpoint_integrity_sha256",
            "log_chain_sha256",
            "trainer_setup_seconds",
            "training_wall_seconds",
            "wall_seconds",
            "files",
        },
        "Exact fit completion schema",
    )
    files = {"initial-weights.pt", "weights.pt", "checkpoint.pt", "training.jsonl"}
    require(set(done["files"]) == files, "Exact fit members")
    for name, digest in done["files"].items():
        checked(folder / name, digest)
    require(all(done[key] == row[key] for key in ("name", "arm", "pair")), "Fit membership")
    checkpoint = load_tensors(folder / "checkpoint.pt")
    body = audit_checkpoint(plan, checkpoint, initial, order, row["arm"], data_hash)
    initial_weights, weights = (
        load_tensors(folder / "initial-weights.pt"),
        load_tensors(folder / "weights.pt"),
    )
    require(
        canonical_tensor_hash(initial_weights) == initial["tensor_hashes"][row["initialization_family"]],
        "Paired initial weights",
    )
    final_hash = canonical_tensor_hash(weights)
    require(
        final_hash == canonical_tensor_hash(body["student_state"]) == done["student_tensor_sha256"],
        "Final deployment weights",
    )
    require(
        done["model_configuration"] == model_configuration(plan, row["arm"])
        and done["settings"] == training_configuration(plan)
        and done["updates"] == done["optimizer_steps"] == body["successful_updates"]
        and done["initialization_sha256"] == initial["integrity_sha256"]
        and done["orders_sha256"] == order["integrity_sha256"]
        and done["data_sha256"] == data_hash
        and done["checkpoint_integrity_sha256"] == checkpoint["integrity_sha256"]
        and done["log_chain_sha256"] == body["log_chain_sha256"]
        and done["parameters"] == sum(math.prod(shape) for shape in shapes(plan, row["arm"]).values()),
        "Fit receipt bindings",
    )
    logs = audit_logs(plan, folder / "training.jsonl", row["arm"], order, body, packets)
    require(
        done["trainer_setup_seconds"] == body["setup_wall_seconds"]
        and done["training_wall_seconds"] == body["training_wall_seconds"]
        and done["trainer_setup_seconds"] + done["training_wall_seconds"]
        <= finite(done["wall_seconds"], "Fit wall", positive=True) + 1e-6,
        "Complete fit cost accounting",
    )
    return {
        "arm": row["arm"],
        "pair": row["pair"],
        "model_configuration": done["model_configuration"],
        "parameters": done["parameters"],
        "student_tensor_sha256": final_hash,
        "wall_seconds": done["wall_seconds"],
        "setup_seconds": done["trainer_setup_seconds"],
        "training_seconds": done["training_wall_seconds"],
        **logs,
    }


def state_shapes(plan, arm):
    result = {"packet": (8,)}
    if arm != "packet_mlp":
        result["hidden"] = (plan["hidden_size"],)
    if arm == "bounded_gru":
        result.update(
            real_packets=(3, 8),
            real_actions=(2, 2),
            real_valid=(3,),
            pending_action=(2,),
            imagined_depth=(1,),
        )
    return result


def model_identity(plan, arm, fit):
    return {
        "version": CONTROL_VERSION,
        "kind": arm,
        "model_class": protocol.MODEL_CLASSES[arm],
        "width": plan["mlp_width"] if arm == "packet_mlp" else plan["hidden_size"],
        "dt": plan["dt"],
        "noise_std": plan["noise_std"],
        "residual_reward": True,
        "real_assimilation": SEMANTICS[arm],
        "weight_tensor_sha256": fit["student_tensor_sha256"],
        "parameter_count": fit["parameters"],
        "parameter_tensor_bytes": fit["parameters"] * 4,
    }


def audit_states(plan, folder, records, arm, fit):
    n, t = plan["control_episodes"], plan["steps"]
    schema = state_shapes(plan, arm)
    values = base.load_npz(
        folder / "states.npz", {f"{phase}__{name}" for phase in ("root", "carried") for name in schema}
    )
    for phase in ("root", "carried"):
        for name, shape in schema.items():
            dtype = {"real_valid": np.bool_, "imagined_depth": np.int64}.get(name, np.float32)
            array(values[f"{phase}__{name}"], (n, t, *shape), dtype, "State schema")
    public, commands = base.stack(records, "policy", "packets"), base.stack(records, "policy", "commands")
    predicted = base.load_npz(folder / "executed_predictions.npz", {"angles", "rewards"})
    require(
        np.array_equal(values["root__packet"], public[:, :t]), "Root packet is exactly current public input"
    )
    next_packet = np.concatenate(
        (
            predicted["angles"],
            public[:, :t, 4:6],
            np.zeros((n, t, 1), np.float32),
            public[:, :t, 7:8] + np.float32(plan["dt"]),
        ),
        axis=-1,
    )
    require(
        np.array_equal(values["carried__packet"], next_packet), "Carried predicted packet/action alignment"
    )
    if arm == "current_gru":
        require(
            np.all(values["root__hidden"][public[:, :t, 6] == 0] == 0),
            "Current-only missing packet resets hidden",
        )
    if arm == "residual_gru":
        missing = public[:, 1:t, 6] == 0
        require(
            np.array_equal(
                values["root__hidden"][:, 1:][missing], values["carried__hidden"][:, :-1][missing]
            ),
            "Missing measurements preserve residual hidden state",
        )
    if arm == "bounded_gru":
        for step in range(t):
            width = min(3, step + 1)
            packets = np.zeros((n, 3, 8), np.float32)
            packets[:, -width:] = public[:, step + 1 - width : step + 1]
            actions = np.zeros((n, 2, 2), np.float32)
            if width > 1:
                actions[:, -(width - 1) :] = commands[:, step + 1 - width : step]
            valid = np.zeros((n, 3), bool)
            valid[:, -width:] = True
            for phase in ("root", "carried"):
                for name, wanted in (
                    ("real_packets", packets),
                    ("real_actions", actions),
                    ("real_valid", valid),
                ):
                    require(
                        np.array_equal(values[f"{phase}__{name}"][:, step], wanted),
                        "Bounded real public packet/action window",
                    )
        require(
            np.all(values["root__pending_action"] == 0)
            and np.all(values["root__imagined_depth"] == 0)
            and np.all(values["carried__imagined_depth"] == 1)
            and np.array_equal(values["carried__pending_action"], commands),
            "Real root and carried one-action phase",
        )
    meta = read(folder / "state-work.json")
    require(
        set(meta)
        == {
            "model",
            "steps",
            "state_arrays_bytes",
            "max_single_candidate_state_tensor_bytes",
            "tensor_bytes_are_not_peak_process_memory",
            "aggregate_model_work",
            "auxiliary_teacher_calls",
            "reset_calls",
        }
        and meta["model"] == model_identity(plan, arm, fit)
        and meta["auxiliary_teacher_calls"] == meta["reset_calls"] == 0
        and meta["tensor_bytes_are_not_peak_process_memory"] is True
        and meta["state_arrays_bytes"] == sum(v.nbytes for v in values.values())
        and len(meta["steps"]) == t,
        "State metadata and payload accounting",
    )
    imagined = 0
    maximum = 0
    for step, row in enumerate(meta["steps"]):
        check_budget()
        roots = {name: torch.from_numpy(values["root__" + name][:, step].copy()) for name in schema}
        carried = {name: torch.from_numpy(values["carried__" + name][:, step].copy()) for name in schema}
        root_bytes = sum(v.numel() * v.element_size() for v in roots.values())
        require(
            row["step"] == step
            and row["root_sha256"] == canonical_tensor_hash(roots)
            and row["carried_sha256"] == canonical_tensor_hash(carried)
            and row["root_tensor_bytes"] == row["carried_tensor_bytes"] == root_bytes,
            "Saved state hashes/bytes",
        )
        search = row["search"]
        transitions = n * 256 * min(plan["planning_horizon"], t - step)
        require(
            search["candidate_evaluations"] == n * 256
            and search["imagined_transitions"] == transitions
            and search["root_tensor_bytes"] == root_bytes
            and search["max_single_candidate_state_tensor_bytes"] == root_bytes * 64
            and search["tensor_bytes_are_not_peak_process_memory"] is True,
            "Candidate state memory/counts",
        )
        audit_work(plan, arm, search["model_work"], 0, transitions)
        audit_work(plan, arm, row["model_work"], n, n + transitions)
        imagined += transitions
        maximum = max(maximum, root_bytes * 64)
    require(meta["max_single_candidate_state_tensor_bytes"] == maximum, "Maximum candidate state payload")
    audit_work(plan, arm, meta["aggregate_model_work"], n * t, n * t + imagined)
    return {
        "actual_class": protocol.MODEL_CLASSES[arm],
        "public_state_buffers_checked": True,
        "state_arrays_bytes": meta["state_arrays_bytes"],
        "maximum_candidate_state_payload_bytes": maximum,
        "model_work": meta["aggregate_model_work"],
        "limits": "Public-buffer/phase arithmetic checked; learned hidden values are not recomputed.",
    }


def audit_prediction(plan, records, values, meta, arm, fit):
    require(
        set(meta)
        == {
            "wall_seconds",
            "student_assimilations",
            "prefix_advances",
            "imagined_advances",
            "teacher_or_auxiliary_calls",
            "model",
            "model_work",
            "retained_root_state_tensor_bytes",
            "output_array_bytes",
            "tensor_bytes_are_not_peak_process_memory",
        },
        "Exact prediction metadata schema",
    )
    metrics = previous.prediction_metrics(plan, records, values)
    n, t = plan["prediction_episodes"], plan["steps"]
    imag = n * sum(min(max(plan["prediction_horizons"]), t - i) for i in range(t))
    for key, count in (
        ("student_assimilations", n * t),
        ("prefix_advances", n * t),
        ("imagined_advances", imag),
        ("teacher_or_auxiliary_calls", 0),
    ):
        exact_integer(meta[key], count, "Prediction work")
    schema = state_shapes(plan, arm)
    root_bytes = n * sum(
        math.prod(shape) * ({"real_valid": 1, "imagined_depth": 8}.get(name, 4))
        for name, shape in schema.items()
    )
    require(
        meta["model"] == model_identity(plan, arm, fit)
        and meta["retained_root_state_tensor_bytes"] == root_bytes * t
        and meta["output_array_bytes"] == sum(v.nbytes for v in values.values())
        and meta["tensor_bytes_are_not_peak_process_memory"] is True,
        "Prediction identity and memory accounting",
    )
    audit_work(plan, arm, meta["model_work"], n * t, n * t + imag)
    return {
        **metrics,
        "timing": {**meta, "wall_seconds": finite(meta["wall_seconds"], "Prediction wall", positive=True)},
    }


def audit_cohort(plan, records, split, panel):
    require(len(records) == plan[f"{split}_episodes"], "Complete fresh cohort")
    transitions, maximum = 0, 0.0
    for i, record in enumerate(records):
        check_budget()
        meta = record["metadata"]
        require(
            meta["seed"] == protocol.seed(plan, f"{split}/reset/{i}")
            and meta["noise_seed"] == protocol.seed(plan, f"{split}/actuator_noise/{i}")
            and meta["noise_std"] == plan["noise_std"]
            and meta["policy_keys"] == ["packets", "commands"]
            and meta["sensor_schedule"] == protocol.schedule(plan, i, panel, split).tolist(),
            "Fresh cohort seed/sensing/public boundary",
        )
        if split == "prediction":
            seed = protocol.seed(plan, f"prediction/exploration/{i}")
            mode = str(
                np.random.default_rng(seed).choice(
                    ["ik_pd", "random_low", "random_high"], p=[0.5, 0.25, 0.25]
                )
            )
            require(
                meta["action_seed"] == seed
                and meta["requested_policy"] == "mixed"
                and meta["collector_policy"] == mode
                and meta["action_hold"] == 4
                and meta["exploration_std"] == {"ik_pd": 0.12, "random_low": 0.3, "random_high": 0.8}[mode],
                "Prediction collector identity",
            )
        replay = search_audit.native.native_replay(record)
        require(
            replay["transitions"] == plan["steps"]
            and replay["new_policy_calls"] == 0
            and replay["saved_output_only"] is True,
            "Native replay scope",
        )
        transitions += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": transitions, "max_abs_error": maximum}


def control_qualification(plan, controls):
    require(
        plan["criterion"] == protocol.CRITERION and set(controls) == set(protocol.PANELS),
        "Frozen memory continuation criteria",
    )
    expected = {row["name"] for row in protocol.fit_manifest(plan)} | set(protocol.REFERENCES)
    for rows in controls.values():
        require(set(rows) == expected, "All paired fits and references retained")
        for row in rows.values():
            value = np.asarray(row["episode_costs"], dtype=float)
            require(
                value.shape == (plan["control_episodes"],)
                and np.isfinite(value).all()
                and math.isclose(float(value.mean()), row["mean_cost"], rel_tol=1e-12, abs_tol=1e-12),
                "Complete paired native cost arithmetic",
            )
    checks, differences = [], {panel: {} for panel in protocol.PANELS}

    def mean(panel, arm):
        return np.mean([controls[panel][f"{arm}-{pair}"]["episode_costs"] for pair in protocol.PAIRS], axis=0)

    def add(name, left, right, comparison="le"):
        checks.append(
            {
                "name": name,
                "left": float(left),
                "right": float(right),
                "comparison": comparison,
                "passed": bool(left < right if comparison == "lt" else left <= right),
            }
        )

    for panel in ("ordinary", "shift"):
        residual = mean(panel, "residual_gru")
        current = mean(panel, "current_gru")
        add(f"{panel}: persistent mean improves current by 5%", residual.mean(), 0.95 * current.mean())
        for pair in protocol.PAIRS:
            add(
                f"{panel}/{pair}: persistent strictly beats current",
                controls[panel][f"residual_gru-{pair}"]["mean_cost"],
                controls[panel][f"current_gru-{pair}"]["mean_cost"],
                "lt",
            )
        add(f"{panel}: persistent mean nonworse than MLP", residual.mean(), mean(panel, "packet_mlp").mean())
        zero = controls[panel]["zero"]["mean_cost"]
        for pair in protocol.PAIRS:
            add(
                f"{panel}/{pair}: persistent beats zero by 10%",
                controls[panel][f"residual_gru-{pair}"]["mean_cost"],
                0.90 * zero,
            )
        for reference in ("known_state", "particle"):
            add(
                f"{panel}/{reference}: physics beats zero by 10%",
                controls[panel][reference]["mean_cost"],
                0.90 * zero,
            )
    add(
        "shift: persistent mean improves bounded by 3%",
        mean("shift", "residual_gru").mean(),
        0.97 * mean("shift", "bounded_gru").mean(),
    )
    for pair in protocol.PAIRS:
        add(
            f"shift/{pair}: persistent nonworse than bounded",
            controls["shift"][f"residual_gru-{pair}"]["mean_cost"],
            controls["shift"][f"bounded_gru-{pair}"]["mean_cost"],
        )
    add(
        "full: persistent mean degrades current at most 2%",
        mean("full", "residual_gru").mean(),
        1.02 * mean("full", "current_gru").mean(),
    )
    for panel in protocol.PANELS:
        for arm in ("current_gru", "bounded_gru", "packet_mlp"):
            differences[panel][f"persistent_minus_{arm}"] = (
                mean(panel, "residual_gru") - mean(panel, arm)
            ).tolist()
        differences[panel]["persistent_minus_public_kinematic"] = (
            mean(panel, "residual_gru") - np.asarray(controls[panel]["public_kinematic"]["episode_costs"])
        ).tolist()
    require(len(checks) == 25, "All25 prescribed checks")
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "public_kinematic": "descriptive_only",
    }, differences


def paired_comparisons(plan, differences):
    result = {}
    for panel, rows in differences.items():
        result[panel] = {}
        for name, items in rows.items():
            values = np.asarray(items, dtype=float)
            rng = np.random.default_rng(protocol.seed(plan, "analysis/bootstrap/0"))
            indices = rng.integers(0, len(values), (plan["bootstrap_samples"], len(values)))
            result[panel][name] = {
                "mean_cost_difference": float(values.mean()),
                "episode_paired_percentile_95": np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist(),
                "cases": len(values),
                "conditional_on_all_three_saved_fit_pairs": True,
            }
    return result


def validate_lineage(plan, execution):
    lineage = plan["objective_source"]
    require(
        set(lineage)
        == {
            "plan_path",
            "plan_sha256",
            "audit_path",
            "audit_receipt_sha256",
            "prior_costs",
            "training_weights_reused",
            "previous_scientific_gate_passed",
        },
        "Objective lineage schema",
    )
    require(
        lineage["plan_sha256"] == "dca778153230375a5a0e790390d328d69857dd4c2d3260b61fb9f214fa1d32f2"
        and lineage["audit_receipt_sha256"]
        == "ee5f68ba9c194d39695c16d257bb05e3f294488a14c025e50a9e881ce4280dc8",
        "Exact completed objective parent",
    )
    parent = read(checked(ROOT / lineage["plan_path"], lineage["plan_sha256"]))
    receipt_path = checked(ROOT / lineage["audit_path"], lineage["audit_receipt_sha256"])
    receipt = read(receipt_path)
    require(
        receipt["status"] == "completed"
        and receipt["saved_output_only"] is True
        and receipt["plan_sha256"] == lineage["plan_sha256"]
        and receipt["source_sha256"] == parent["sources"]
        and receipt["costs"] == lineage["prior_costs"]
        and lineage["training_weights_reused"] is False
        and lineage["previous_scientific_gate_passed"] is False,
        "Completed objective identity; no reused weights",
    )
    for name, digest in receipt["files"].items():
        checked(receipt_path.parent / name, digest)
    for name, digest in parent["sources"].items():
        checked(ROOT / name, digest)
    require(plan["training_source"] == parent["training_source"], "Unchanged inherited training source")
    cohort, _ = previous.validate_lineage(parent, execution)
    return cohort, parent


def validate_runtime_sources(plan):
    previous.validate_runtime_sources(plan)


def validate_streams(plan, parent, execution):
    # Reconstruct the historical registries from the authenticated completed
    # parent's descriptors, then append the objective study and its engineering exclusions.
    contract = parent["random_stream_contract"]
    descriptors = copy.deepcopy(contract["priors"])
    numpy_prior = []
    torch_prior = {}
    for prefix, seeds in (("world", (211, 223, 239)), ("reward", (271, 283, 293))):
        for seed in seeds:
            torch_prior[f"{prefix}/student/{seed}"] = seed
            torch_prior[f"{prefix}/minibatch/{seed}"] = seed + 4_100_000
    for descriptor in descriptors:
        kind = descriptor.get("registry_kind")
        if kind is None:
            source = read(checked(ROOT / descriptor["plan_path"], descriptor["plan_sha256"]))
            numpy_prior.append(concrete_streams(source, include_train=descriptor["include_train"]))
        elif kind == "named-search":
            source = read(checked(ROOT / descriptor["plan_path"], descriptor["plan_sha256"]))
            numpy_prior.append(search_audit.protocol.registry(source))
        elif kind == "named-search-engineering":
            source = read(checked(ROOT / descriptor["plan_path"], descriptor["plan_sha256"]))
            exclusions = [
                row
                for row in source["random_stream_contract"]["engineering_exclusions"]
                if row["namespace"] == descriptor["namespace"]
            ]
            require(len(exclusions) == 1, "Historical engineering namespace")
            numpy_prior.append(exclusions[0]["registry"])
        elif kind in ("additional-engineering-numpy", "additional-engineering-torch"):
            path = checked(ROOT / descriptor["artifact_path"], descriptor["artifact_sha256"])
            if "seeds" in descriptor:
                seeds = descriptor["seeds"]
            else:
                capacity = read(path)
                seeds = {
                    "engineering/capacity/synthetic/public_data": capacity["synthetic_data"]["seed"],
                    "engineering/temporary-overwritten-construction": 0,
                }
                actual = protocol.torch_generator_manifest({"capacity": capacity["synthetic_data"]["seed"]})
                require(
                    actual["capacity"]["initial_state_sha256"]
                    == capacity["synthetic_data"]["initial_state_sha256"],
                    "Historical capacity RNG",
                )
            if kind.endswith("numpy"):
                numpy_prior.append(seeds)
            else:
                torch_prior.update(seeds)
        else:
            raise ValueError("Unknown authenticated historical RNG descriptor")
    require(
        [protocol._identity(row) for row in numpy_prior] == contract["prior_numpy_registry_sha256"]
        and [protocol._identity(torch_prior)] == contract["prior_torch_registry_sha256"],
        "Historical prior registry reconstruction",
    )
    numpy_prior.append(contract["registry"])
    torch_registries = [torch_prior, contract["torch_registry"]]
    lineage = plan["objective_source"]
    descriptors.append(
        {
            "plan_path": lineage["plan_path"],
            "plan_sha256": lineage["plan_sha256"],
            "registry_kind": "objective-scored",
        }
    )
    for row in contract["engineering_exclusions"]:
        numpy_prior.append(row["registry"])
        torch_registries.append(row["torch_registry"])
        descriptors.append(
            {
                "plan_path": lineage["plan_path"],
                "plan_sha256": lineage["plan_sha256"],
                "namespace": row["namespace"],
                "registry_kind": "objective-engineering",
            }
        )
    numpy_prior.append({"memory/engineering/fixture": 410})
    torch_registries.append({"memory/engineering/fixture": 410, "memory/engineering/restore": 0})
    descriptors.append(
        {"registry_kind": "memory-engineering-literal", "numpy_seeds": [410], "torch_seeds": [0, 410]}
    )
    expected = protocol.stream_contract(plan, numpy_prior, torch_registries, descriptors)
    require(
        plan["random_stream_contract"] == expected and read(execution / "random-streams.json") == expected,
        "Complete prior/engineering stream exclusions",
    )
    return {
        "numpy_generators": len(expected["generators"]),
        "torch_generators": len(expected["torch_generators"]),
        "prior_numpy_registries": len(numpy_prior),
        "prior_torch_registries": len(torch_registries),
        "draws_for_manifest": 0,
    }


def expected_members(plan):
    members = {
        "started.json",
        "random-streams.json",
        "train.npz",
        "train.json",
        "train-provenance.json",
        "training-prepared.json",
        "all-fits-completed.json",
        "restored-models.json",
        "evaluation-started.json",
        "prediction.npz",
        "prediction.json",
        "costs.json",
    }
    for pair in protocol.PAIRS:
        members.update(
            f"{folder}/{pair}.{suffix}"
            for folder in ("initializations", "orders")
            for suffix in ("pt", "json")
        )
    for row in protocol.fit_manifest(plan):
        members.update(
            f"fits/{row['name']}/{name}"
            for name in (
                "initial-weights.pt",
                "weights.pt",
                "checkpoint.pt",
                "training.jsonl",
                "completed.json",
            )
        )
        members.update(f"predictions/{row['name']}.{suffix}" for suffix in ("npz", "json"))
    for step in range(plan["steps"]):
        members.update(f"innovations/control/{step:03d}.{suffix}" for suffix in ("npz", "json"))
    for row in protocol.execution_order(plan):
        path = row["path"]
        members.update(f"{path}/{name}" for name in ("episodes.npz", "episodes.json", "timings.json"))
        if "fit" in row:
            members.update(
                f"{path}/{name}" for name in ("executed_predictions.npz", "states.npz", "state-work.json")
            )
            members.update(
                f"{path}/decisions/{step:03d}.{suffix}"
                for step in range(plan["steps"])
                for suffix in ("npz", "json")
            )
        else:
            members.add(f"{path}/planning.npz")
            if row["reference"] == "public_kinematic":
                members.add(f"{path}/observer-final.json")
    return members


def validate_members(plan, expected, execution):
    require(
        isinstance(expected, str) and len(expected) == 64 and set(expected) <= set("0123456789abcdef"),
        "External frozen plan SHA",
    )
    completed = read(execution / "completed.json")
    require(
        set(completed)
        == {
            "status",
            "study",
            "plan_sha256",
            "fits",
            "control_rows",
            "prediction_episodes",
            "astra_calls",
            "wall_seconds",
            "evaluation_started_elapsed_seconds",
            "cumulative_attempt_wall_seconds",
            "files",
        }
        and completed["status"] == "completed"
        and completed["study"] == VERSION
        and completed["plan_sha256"] == expected,
        "Complete memory execution identity",
    )
    for key, value in (
        ("fits", 12),
        ("control_rows", 51),
        ("prediction_episodes", plan["prediction_episodes"]),
        ("astra_calls", 0),
    ):
        exact_integer(completed[key], value, "Completed scope " + key)
    require(
        finite(completed["wall_seconds"], "Whole execution wall", positive=True) <= plan["cap_seconds"],
        "Frozen execution cap",
    )
    paths = list(execution.rglob("*"))
    require(
        not execution.is_symlink() and not any(path.is_symlink() for path in paths), "No artifact symlinks"
    )
    members = expected_members(plan)
    require(
        {path.relative_to(execution).as_posix() for path in paths if path.is_file()}
        == members | {"completed.json"}
        and set(completed["files"]) == members,
        "Exact execution members; no partial/failed attempts",
    )
    for name, digest in completed["files"].items():
        checked(execution / name, digest)
    return completed


def audit_boundaries(plan, expected, execution, completed, fits):
    started, prepared, all_fits, restored, evaluation = (
        read(execution / name)
        for name in (
            "started.json",
            "training-prepared.json",
            "all-fits-completed.json",
            "restored-models.json",
            "evaluation-started.json",
        )
    )
    for value, keys in (
        (started, {"plan_sha256", "unix_time"}),
        (prepared, {"files", "optimizer_updates", "wall_seconds", "unix_time"}),
        (
            all_fits,
            {"plan_sha256", "fit_order", "files", "training_prepared_sha256", "unix_time", "elapsed_seconds"},
        ),
        (restored, {"models", "wall_seconds", "unix_time"}),
        (
            evaluation,
            {
                "plan_sha256",
                "all_fits_completed_sha256",
                "restored_models_sha256",
                "unix_time",
                "elapsed_seconds",
            },
        ),
    ):
        require(set(value) == keys, "Exact phase boundary schema")
    prepfiles = {
        f"{folder}/{pair}.{suffix}": sha(execution / folder / f"{pair}.{suffix}")
        for folder in ("initializations", "orders")
        for pair in protocol.PAIRS
        for suffix in ("pt", "json")
    }
    require(
        prepared["files"] == prepfiles and prepared["optimizer_updates"] == 0,
        "All initialization and orders precede fitting",
    )
    require(
        started["plan_sha256"] == all_fits["plan_sha256"] == evaluation["plan_sha256"] == expected
        and all_fits["fit_order"] == plan["fit_order"]
        and all_fits["files"]
        == {
            f"fits/{name}/completed.json": sha(execution / "fits" / name / "completed.json")
            for name in plan["fit_order"]
        }
        and all_fits["training_prepared_sha256"] == sha(execution / "training-prepared.json"),
        "All12 fits authenticated before restore",
    )
    require(set(restored["models"]) == set(plan["fit_order"]), "All12 actual models restored")
    for name, fit in fits.items():
        require(
            restored["models"][name]
            == {
                "kind": fit["arm"],
                "model_class": protocol.MODEL_CLASSES[fit["arm"]],
                "checkpoint_sha256": sha(execution / "fits" / name / "checkpoint.pt"),
                "weights_sha256": sha(execution / "fits" / name / "weights.pt"),
                "student_tensor_sha256": fit["student_tensor_sha256"],
                "successful_updates": fit["updates"],
                "configuration": fit["model_configuration"],
            },
            "Restored class, weights and semantics",
        )
    require(
        evaluation["all_fits_completed_sha256"] == sha(execution / "all-fits-completed.json")
        and evaluation["restored_models_sha256"] == sha(execution / "restored-models.json")
        and evaluation["elapsed_seconds"] == completed["evaluation_started_elapsed_seconds"],
        "Fresh evaluation boundary identity",
    )
    dates = [
        finite(row["unix_time"], "Phase Unix timestamp", positive=True)
        for row in (started, prepared, all_fits, restored, evaluation)
    ]
    require(
        dates == sorted(dates)
        and 0 < all_fits["elapsed_seconds"] < evaluation["elapsed_seconds"] < completed["wall_seconds"],
        "Causal fit/restore/evaluate order",
    )
    restore = finite(restored["wall_seconds"], "Restoration wall", positive=True)
    require(
        all_fits["elapsed_seconds"] + restore <= evaluation["elapsed_seconds"] + 1e-6,
        "Charged pre-evaluation restoration",
    )
    return {
        "all_twelve_fits_restored_before_evaluation": True,
        "training_preparation_seconds": finite(prepared["wall_seconds"], "Preparation wall", positive=True),
        "restore_seconds": restore,
        "fit_completed_elapsed_seconds": all_fits["elapsed_seconds"],
        "evaluation_started_elapsed_seconds": evaluation["elapsed_seconds"],
        "scope": "Hash-bound source/log ordering, not an independent process observer.",
    }


def audit_inherited_training(cohort_plan, records):
    return base.audit_cohort(cohort_plan, records, "train")


def public_learning_tensors(records):
    """Independent allowlist matching the fit's public inputs and reward labels."""
    return {
        name: torch.tensor(base.stack(records, group, name), dtype=torch.float32)
        for group, name in (("policy", "packets"), ("policy", "commands"), ("audit", "rewards"))
    }


def kinematic_configuration(model, plan):
    return {
        "version": "reacher-public-two-measurement-kinematics-v1",
        "classification": "supplied-physics public-observation competence reference",
        "learned": False,
        "privileged_live_state": False,
        "supplied_knowledge": "Full copied Reacher MjModel geometry, masses, actuators, joint limits and dynamics",
        "initial_velocity": "exactly zero, including target coordinates",
        "initial_angle_branch": "principal atan2; initial winding is unidentifiable",
        "measurement_velocity": "wrapped shortest displacement / actual time since previous valid packet",
        "measurement_position": "nearest nominally predicted angle branch, including soft-limit crossings",
        "missing_angles": "finite zero placeholders required; never use unobserved angles",
        "target": "static public target; target velocity zero",
        "disturbance_model": "nominal zero disturbance; realized noise unavailable",
        "command": "clip [-1,1], then float32 quantization before nominal native propagation",
        "frame_skip": 2,
        "native_timestep": float(model.opt.timestep),
        "decision_dt": plan["dt"],
        "integrator": int(model.opt.integrator),
        "gravity": model.opt.gravity.tolist(),
        "measurement_limit": "interval-average velocity; displacement exceeding pi can alias",
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def audit_kinematic_observers(plan, records, estimates, snapshots):
    """Reconstruct only public-derived state, never rerun MPC or a learned model.

    The final real packet is not assimilated: the last planning root is step49.
    Nominal observer propagation is additional audit work, counted separately
    from replay of the saved disturbed control episodes.
    """
    native = search_audit.native
    require(
        isinstance(snapshots, list) and len(snapshots) == len(records), "Complete public observer snapshots"
    )
    env = native.make_env()
    transitions, maximum = 0, 0.0
    try:
        model = copy.copy(env.unwrapped.model)
        require(float(model.opt.timestep) * 2 == plan["dt"], "Nominal observer decision time")
        steps, dt = plan["steps"], plan["dt"]
        planned = 64 * sum(min(plan["planning_horizon"], steps - t) for t in range(steps))
        for index, (record, saved) in enumerate(zip(records, snapshots, strict=True)):
            check_budget()
            packets = np.asarray(record["policy"]["packets"], dtype=np.float64)
            commands = record["policy"]["commands"]
            data = native.mujoco.MjData(model)
            target = packets[0, 4:6].copy()
            last_angles = np.arctan2(packets[0, 2:4], packets[0, :2])
            data.qpos[:] = np.concatenate((last_angles, target))
            data.qvel[:] = 0.0
            native.mujoco.mj_forward(model, data)
            last_valid, interval, visible = 0, None, [0]
            for step in range(steps):
                check_budget()
                packet = packets[step]
                require(np.array_equal(packet[4:6], target), "Observer uses a static public target")
                if step:
                    data.ctrl[:] = commands[step - 1].astype(np.float64)
                    native.mujoco.mj_step(model, data, nstep=2)
                    native.mujoco.mj_rnePostConstraint(model, data)
                    transitions += 1
                    elapsed = (step - last_valid) * dt
                    require(
                        np.isclose(packet[7], 0.0 if packet[6] else elapsed, atol=1e-6, rtol=1e-6),
                        "Public elapsed age",
                    )
                    if packet[6]:
                        angles = np.arctan2(packet[2:4], packet[:2])
                        data.qvel[:2] = ((angles - last_angles + np.pi) % (2 * np.pi) - np.pi) / elapsed
                        data.qvel[2:] = 0.0
                        data.qpos[:2] += (angles - data.qpos[:2] + np.pi) % (2 * np.pi) - np.pi
                        data.qpos[2:] = target
                        native.mujoco.mj_forward(model, data)
                        last_angles, last_valid, interval = angles.copy(), step, elapsed
                        visible.append(step)
                expected = np.concatenate((data.qpos, data.qvel))
                error = float(np.max(np.abs(estimates[index, step] - expected)))
                require(math.isfinite(error) and error <= 1e-10, "Public-only nominal state reconstruction")
                maximum = max(maximum, error)
            require(
                set(saved) == {"observer", "failed", "planner", "reward_weights", "costs"}
                and saved["failed"] is False
                and saved["planner"]
                == "Existing PhysicsMPC; supplied physics and public-derived estimate, not true state"
                and saved["reward_weights"] == {"distance": 1.0, "control": 1.0},
                "Public reference identity",
            )
            observer = saved["observer"]
            require(
                set(observer)
                == {
                    "configuration",
                    "step_index",
                    "elapsed_seconds",
                    "last_valid_step",
                    "last_velocity_interval_seconds",
                    "valid_measurements",
                    "last_packet",
                    "last_issued_command",
                    "qpos_estimate",
                    "qvel_estimate",
                    "failed",
                    "costs",
                },
                "Observer snapshot schema",
            )
            require(
                observer["configuration"] == kinematic_configuration(model, plan)
                and observer["step_index"] == steps - 1
                and observer["elapsed_seconds"] == (steps - 1) * dt
                and observer["last_valid_step"] == last_valid
                and observer["last_velocity_interval_seconds"] == interval
                and observer["failed"] is False,
                "Final observer time, visibility and semantics",
            )
            require(
                observer["valid_measurements"]
                == [{"step": t, "time": t * dt, "packet": packets[t].tolist()} for t in visible[-2:]],
                "Actual last two public measurements",
            )
            for name, wanted in (
                ("last_packet", packets[steps - 1]),
                ("last_issued_command", commands[steps - 2]),
                ("qpos_estimate", data.qpos),
                ("qvel_estimate", data.qvel),
            ):
                actual = np.asarray(observer[name], dtype=np.float64)
                require(
                    actual.shape == wanted.shape
                    and np.isfinite(actual).all()
                    and np.allclose(actual, wanted, atol=1e-10, rtol=0),
                    "Final observer field: " + name,
                )
            oc = observer["costs"]
            expected_counts = {
                "native_transition_attempts": steps - 1,
                "native_substeps_requested": 2 * (steps - 1),
                "native_transitions_completed": steps - 1,
                "native_substeps_completed": 2 * (steps - 1),
                "setup_forward_calls": 1,
                "measurement_reanchor_forward_calls": len(visible) - 1,
                "postconstraint_refresh_calls_completed": steps - 1,
            }
            require(
                set(oc)
                == set(expected_counts)
                | {
                    "setup_wall_seconds",
                    "update_wall_seconds",
                    "counts_exclude_forward_and_postconstraint_from_native_substeps",
                }
                and oc["counts_exclude_forward_and_postconstraint_from_native_substeps"] is True,
                "Observer work schema",
            )
            for name, value in expected_counts.items():
                exact_integer(oc[name], value, "Observer " + name)
            for name in ("setup_wall_seconds", "update_wall_seconds"):
                finite(oc[name], "Observer " + name, positive=True)
            pc = saved["costs"]
            expected_counts = {
                "successful_plan_calls": steps,
                "candidate_evaluations_completed": steps * 64,
                "candidate_reset_forward_calls_completed": steps * 64,
                "native_transitions_requested": planned,
                "native_substeps_requested": planned * 2,
                "native_transitions_completed": planned,
                "native_substeps_completed": planned * 2,
                "postconstraint_refresh_calls_completed": planned,
            }
            require(
                set(pc)
                == set(expected_counts)
                | {
                    "setup_wall_seconds_including_observer",
                    "planning_wall_seconds",
                    "setup_and_observer_costs_overlap_do_not_sum",
                    "failed_plan_partial_work_is_bounded_by_requested_counts",
                }
                and pc["setup_and_observer_costs_overlap_do_not_sum"] is True
                and pc["failed_plan_partial_work_is_bounded_by_requested_counts"] is True,
                "Nominal planner work schema",
            )
            for name, value in expected_counts.items():
                exact_integer(pc[name], value, "Nominal planner " + name)
            require(
                finite(pc["setup_wall_seconds_including_observer"], "Planner setup", positive=True)
                >= oc["setup_wall_seconds"],
                "Nested observer setup",
            )
            finite(pc["planning_wall_seconds"], "Planner wall", positive=True)
    finally:
        env.close()
    return {
        "observer_native_transitions_replayed": transitions,
        "max_abs_error": maximum,
        "final_observer_root": plan["steps"] - 1,
        "new_planner_calls": 0,
        "scope": "Public packets and issued commands only; nominal observer replay, no realized noise or true state.",
    }


def audit_kinematic_control(plan, folder, records, inputs):
    n, steps = plan["control_episodes"], plan["steps"]
    times = read(folder / "timings.json")
    require(
        set(times)
        == {
            "setup_seconds",
            "decision_seconds",
            "native_step_seconds",
            "row_wall_seconds",
            "candidate_evaluations_per_decision",
            "information",
        }
        and times["information"] == "public packets, issued commands, supplied nominal model; no true state",
        "Kinematic timing schema",
    )
    exact_integer(times["candidate_evaluations_per_decision"], 64, "Nominal candidate count")
    for name in ("decision_seconds", "native_step_seconds"):
        require(isinstance(times[name], list) and len(times[name]) == steps, "Nominal timing coverage")
        for value in times[name]:
            finite(value, name, positive=True)
    for name in ("setup_seconds", "row_wall_seconds"):
        finite(times[name], name, positive=True)
    require(
        times["setup_seconds"] + sum(times["decision_seconds"]) + sum(times["native_step_seconds"])
        <= times["row_wall_seconds"] + 1e-6,
        "Nominal nested row times",
    )
    values = base.load_npz(folder / "planning.npz", {"candidate_scores", "planner_used", "public_estimates"})
    scores = array(
        values["candidate_scores"], (n, steps, 64), np.float64, "Public reference candidate scores"
    )
    estimates = array(values["public_estimates"], (n, steps, 8), np.float64, "Public state estimates")
    require(
        bool(array(values["planner_used"], (), np.bool_, "Nominal planner flag")), "Nominal planning enabled"
    )
    commands = base.stack(records, "policy", "commands")
    imagined = 0
    for step in range(steps):
        check_budget()
        bank = search_audit.common_bank(plan, inputs[step], step)
        require(
            np.array_equal(commands[:, step], bank[np.arange(n), scores[:, step].argmax(1), 0]),
            "Public reference earliest-best selected action",
        )
        imagined += n * 64 * bank.shape[2]
    snapshots = read(folder / "observer-final.json")
    observers = audit_kinematic_observers(plan, records, estimates, snapshots)
    require(
        sum(v["costs"]["setup_wall_seconds_including_observer"] for v in snapshots)
        <= times["setup_seconds"] + 1e-6
        and sum(
            v["costs"]["planning_wall_seconds"] + v["observer"]["costs"]["update_wall_seconds"]
            for v in snapshots
        )
        <= sum(times["decision_seconds"]) + 1e-6,
        "Nominal observer/planner costs charged inside row",
    )
    costs = -base.stack(records, "audit", "rewards").sum(1)
    decisions = np.asarray(times["decision_seconds"], dtype=float)
    return {
        "episode_costs": costs.tolist(),
        "mean_cost": float(costs.mean()),
        "setup_seconds": times["setup_seconds"],
        "decision_wall_seconds": float(decisions.sum()),
        "native_step_seconds": float(sum(times["native_step_seconds"])),
        "row_wall_seconds": times["row_wall_seconds"],
        "batch_latency_seconds": {
            "mean": float(decisions.mean()),
            "p50": float(np.quantile(decisions, 0.5)),
            "p95": float(np.quantile(decisions, 0.95)),
            "max": float(decisions.max()),
        },
        "per_case_amortized_seconds": float(decisions.mean() / n),
        "decision_seconds": times["decision_seconds"],
        "search_seconds": [],
        "candidate_evaluations": n * steps * 64,
        "imagined_transitions": imagined,
        "clipping_fraction": None,
        "planner_used": True,
        "public_observer": observers,
        "scope": "Supplied-physics public-observation competence reference; descriptive only.",
    }


def audit_costs(plan, execution, completed, initializations, fits, boundary, predictions, controls):
    saved = read(execution / "costs.json")
    phases = {
        "training_preparation_seconds",
        "fit_wall_seconds",
        "restore_wall_seconds",
        "prediction_collection_seconds",
        "prediction_model_seconds",
        "innovation_generation_and_storage_seconds",
        "control_row_wall_seconds",
        "control_setup_seconds",
        "control_decision_seconds",
        "control_native_step_seconds",
    }
    require(
        set(saved) == phases | {"new_fits", "astra_calls", "prior_costs", "compute_matched", "accounting"},
        "Whole memory study cost schema",
    )
    for name in phases:
        finite(saved[name], name, positive=True)
    rows = [row for panel in controls.values() for row in panel.values()]
    computed = {
        "training_preparation_seconds": boundary["training_preparation_seconds"],
        "fit_wall_seconds": sum(row["wall_seconds"] for row in fits.values()),
        "restore_wall_seconds": boundary["restore_seconds"],
        "prediction_model_seconds": sum(row["timing"]["wall_seconds"] for row in predictions.values()),
        "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in rows),
        "control_setup_seconds": sum(row["setup_seconds"] for row in rows),
        "control_decision_seconds": sum(row["decision_wall_seconds"] for row in rows),
        "control_native_step_seconds": sum(row["native_step_seconds"] for row in rows),
    }
    for name, value in computed.items():
        require(
            math.isclose(saved[name], value, rel_tol=1e-12, abs_tol=1e-7), "Reconstructed phase cost: " + name
        )
    require(
        sum(row["wall_seconds"] for row in initializations.values())
        <= saved["training_preparation_seconds"] + 1e-6,
        "Initialization nested in training preparation",
    )
    before_fit = saved["training_preparation_seconds"] + saved["fit_wall_seconds"]
    evaluation = sum(
        saved[name]
        for name in (
            "prediction_collection_seconds",
            "prediction_model_seconds",
            "innovation_generation_and_storage_seconds",
            "control_row_wall_seconds",
        )
    )
    require(
        before_fit <= boundary["fit_completed_elapsed_seconds"] + 1e-6
        and before_fit + saved["restore_wall_seconds"]
        <= boundary["evaluation_started_elapsed_seconds"] + 1e-6
        and boundary["evaluation_started_elapsed_seconds"] + evaluation <= completed["wall_seconds"] + 1e-6,
        "Nonoverlapping phase costs fit the whole-run cap",
    )
    exact_integer(saved["new_fits"], 12, "All12 fits charged")
    exact_integer(saved["astra_calls"], 0, "No remote model calls")
    require(
        saved["compute_matched"] is False
        and isinstance(saved["accounting"], str)
        and bool(saved["accounting"]),
        "Compute mismatch disclosed",
    )
    require(saved["prior_costs"] == plan["objective_source"]["prior_costs"], "Prior cumulative cost identity")
    earlier = finite(saved["prior_costs"]["cumulative_attempt_wall_seconds"], "Earlier attempts")
    require(
        earlier >= 0 and (plan.get("engineering") is True or earlier > 0),
        "Nonnegative engineering or positive authenticated prior cost",
    )
    require(
        completed["cumulative_attempt_wall_seconds"] == earlier + completed["wall_seconds"],
        "Charge new execution exactly once",
    )
    return {
        **saved,
        "new_execution_wall_seconds": completed["wall_seconds"],
        "cumulative_attempt_wall_seconds": completed["cumulative_attempt_wall_seconds"],
    }


def audit_saved(plan, expected_plan_sha256, execution, out, *, engineering=False):
    """Audit saved evidence; caller authenticates external plan bytes.

    Engineering fixtures may replace only inherited provenance/native-cohort
    identity hooks. All fresh artifacts, training tensors, streams, native
    controls and reported metrics still follow the complete audit path.
    """
    execution, out = Path(execution), Path(out)
    require(not out.exists(), "Exclusive memory audit output required")
    begin = time.monotonic()
    require(
        type(plan["audit_cap_seconds"]) is int and plan["audit_cap_seconds"] > 0, "Frozen positive audit cap"
    )
    token = _DEADLINE.set(begin + plan["audit_cap_seconds"])
    try:
        require(type(plan["cap_seconds"]) is int and plan["cap_seconds"] > 0, "Frozen positive execution cap")
        protocol.validate_settings(plan, engineering=engineering)
        require(plan["engineering"] is engineering, "Explicit engineering audit boundary")
        validate_runtime_sources(plan)
        completed = validate_members(plan, expected_plan_sha256, execution)
        completion_hash = sha(execution / "completed.json")
        cohort_plan, parent = validate_lineage(plan, execution)
        require(
            set(plan["sources"]) == set(parent["sources"]) | NEW_SOURCES,
            "Exact frozen implementation membership",
        )
        streams = validate_streams(plan, parent, execution)
        train = base.load_records(execution / "train", plan["train_episodes"])
        replay = {"inherited_training": audit_inherited_training(cohort_plan, train)}
        check_budget()
        data = public_learning_tensors(train)
        data_hash = canonical_tensor_hash(data)
        initializations, orders = audit_preparation(plan, execution)
        fits = {}
        for row in protocol.fit_manifest(plan):
            fits[row["name"]] = audit_fit(
                plan,
                execution,
                row,
                initializations[row["pair"]],
                orders[row["pair"]],
                data_hash,
                data["packets"].numpy(),
            )
        boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, fits)
        prediction = base.load_records(execution / "prediction", plan["prediction_episodes"])
        replay["prediction"] = audit_cohort(plan, prediction, "prediction", "ordinary")
        predictions = {}
        for name in plan["fit_order"]:
            check_budget()
            predictions[name] = audit_prediction(
                plan,
                prediction,
                base.load_npz(execution / "predictions" / f"{name}.npz"),
                read(execution / "predictions" / f"{name}.json"),
                fits[name]["arm"],
                fits[name],
            )
        inputs = [
            search_audit.audit_innovations(
                plan,
                execution / "innovations" / "control" / f"{step:03d}",
                f"planner/control/{step}",
                plan["control_episodes"],
            )
            for step in range(plan["steps"])
        ]
        controls, states, disturbances = {panel: {} for panel in protocol.PANELS}, None, None
        for row in protocol.execution_order(plan):
            check_budget()
            folder = execution / row["path"]
            records = base.load_records(folder / "episodes", plan["control_episodes"])
            replay[row["path"]] = audit_cohort(plan, records, "control", row["panel"])
            current, noise = (
                base.stack(records, "audit", "integration_state")[:, 0],
                base.stack(records, "audit", "actuator_noise"),
            )
            if states is None:
                states, disturbances = current, noise
            require(
                np.array_equal(states, current) and np.array_equal(disturbances, noise),
                "All learned and reference rows share resets and exogenous noise",
            )
            if row.get("reference") == "public_kinematic":
                control = audit_kinematic_control(plan, folder, records, inputs)
            else:
                control = search_audit.audit_control(
                    {**plan, "planners": ["cem256"]}, folder, row, records, inputs
                )
            if "fit" in row:
                fit = fits[row["fit"]]
                control["state_and_work"] = audit_states(plan, folder, records, fit["arm"], fit)
            controls[row["panel"]][row["label"]] = control
        gate, differences = control_qualification(plan, controls)
        costs = audit_costs(
            plan, execution, completed, initializations, fits, boundary, predictions, controls
        )
        comparisons = paired_comparisons(plan, differences)
        require(
            validate_members(plan, expected_plan_sha256, execution) == completed
            and sha(execution / "completed.json") == completion_hash,
            "Execution changed during audit",
        )
        validate_runtime_sources(plan)
        check_budget()
        summary = {
            "status": "completed",
            "version": VERSION,
            "engineering": engineering,
            "plan_sha256": expected_plan_sha256,
            "execution_completed_sha256": completion_hash,
            "saved_output_only": True,
            "new_model_calls": 0,
            "new_policy_calls": 0,
            "new_fits": 0,
            "native_transitions_checked": sum(row["transitions"] for row in replay.values()),
            "native_max_abs_error": max(row["max_abs_error"] for row in replay.values()),
            "public_observer_transitions_checked": sum(
                controls[panel]["public_kinematic"]["public_observer"]["observer_native_transitions_replayed"]
                for panel in protocol.PANELS
            ),
            "coverage": protocol.coverage(plan),
            "random_streams": streams,
            "fits": fits,
            "phase_boundary": boundary,
            "control": controls,
            "prediction": predictions,
            "continuation_gate": gate,
            "paired_descriptive_comparisons": comparisons,
            "cohorts": replay,
            "costs": {**costs, "audit_validation_wall_seconds": time.monotonic() - begin},
            "limits": [
                "Saved final tensors, Adam state, update cursor and log chain are authenticated; neural training and optimizer transitions are not independently rerun.",
                "Actual model class and history semantics bind to configuration and source even when GRU state-dictionary shapes match.",
                "Public history buffers, missing-observation resets and phase transitions are checked; learned hidden values and neural predictions are not recomputed.",
                "Native replay validates actions/outcomes. CEM reconstructs all proposals and earliest-best decisions from saved scores without learned-model calls.",
                "Public kinematics is a supplied-physics reference with zero initial velocity and interval-average wrapped velocities; neither learned nor privileged true-state control.",
                "Equal optimizer updates and paired GRU initialization are not equal training compute. MLP width, actual operations and state payload bytes are reported separately.",
                "All12 fits and51 control rows are retained. Bootstrap intervals condition on three paired fits and one shared training corpus.",
                "This single-environment architecture comparison cannot establish biological, JEPA or general robotics novelty.",
                "A persistent-history advantage could reflect retention of last visible angles or velocity inference; this study does not separate those mechanisms. Public kinematics supplies physics and is not a hold-last-angle ablation.",
            ],
        }
        out.mkdir(parents=True, exist_ok=False)
        write(out / "summary.json", summary)
        (out / "README.md").write_text(
            "# Reacher memory saved-output audit\n\n"
            f"Continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(row['passed'] for row in gate['checks'])}/25 checks).\n\n"
            f"All12 fits and51 control rows retained; {summary['native_transitions_checked']:,} native outcome transitions replayed. "
            f"A further {summary['public_observer_transitions_checked']:,} nominal public-observer transitions were reconstructed.\n\n"
            "No learned model was instantiated or called. Recorded CEM proposals, class/configuration, public history buffers, "
            "checkpoint/optimizer state and cost accounting were checked. Neural training and predictions remain source-bound evidence. "
            "See summary.json for the complete qualification results and limits.\n"
        )
        receipt = {
            "status": "completed",
            "version": VERSION,
            "engineering": engineering,
            "plan_sha256": expected_plan_sha256,
            "source_sha256": plan["sources"],
            "runtime": plan["runtime"],
            "execution_completed_sha256": completion_hash,
            "execution_members": completed["files"],
            "saved_output_only": True,
            "costs": summary["costs"],
            "files": {name: sha(out / name) for name in ("summary.json", "README.md")},
        }
        check_budget()
        write(out / "receipt.json", receipt)
        check_budget()
        return summary
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "receipt.json").exists():
            (out / "receipt.json").rename(out / "over-cap-receipt.json")
        write(
            out / "failed.json",
            {
                "status": "failed",
                "error": repr(error),
                "plan_sha256": expected_plan_sha256,
                "wall_seconds": time.monotonic() - begin,
                "saved_output_only": True,
            },
        )
        raise
    finally:
        _DEADLINE.reset(token)


def audit_plan(path, expected_sha256, execution, out):
    """Authenticate a frozen external plan; CLI never accepts engineering bypasses."""
    out = Path(out)
    require(not out.exists(), "Exclusive memory audit output required")
    try:
        plan = read(checked(path, expected_sha256))
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=False)
        write(
            out / "failed.json",
            {
                "status": "failed",
                "error": repr(error),
                "plan_sha256": expected_sha256,
                "stage": "plan_authentication",
                "saved_output_only": True,
            },
        )
        raise
    return audit_saved(plan, expected_sha256, execution, out, engineering=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = audit_plan(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(
        json.dumps({"status": value["status"], "continuation_passed": value["continuation_gate"]["passed"]})
    )


if __name__ == "__main__":
    main()
