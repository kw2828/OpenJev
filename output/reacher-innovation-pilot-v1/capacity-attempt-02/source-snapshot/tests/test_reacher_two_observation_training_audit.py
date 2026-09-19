"""Handbuilt tensor/log payloads; no model, optimizer, native or production data."""

import ast
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

import pytest
import torch

from openjev.research import reacher_two_observation_training_audit as audit


def seal(body):
    body.pop("integrity_sha256", None)
    body["integrity_sha256"] = audit.state_hash(body)
    return body


def rechain(case):
    previous = audit.state_hash([])
    for row in case["logs"]:
        row["previous_log_sha256"] = previous
        row.pop("log_sha256", None)
        row["log_sha256"] = audit.state_hash(row)
        previous = row["log_sha256"]
    case["checkpoint"]["log_chain_sha256"] = previous
    seal(case["checkpoint"])
    case["expected_checkpoint_sha256"] = case["checkpoint"]["integrity_sha256"]


def fixture():
    # Literal410 is an existing excluded engineering seed. Only historical
    # permutation payloads are constructed; weights/Adam/losses are fabricated.
    settings = {"hidden_size": 2, "mlp_width": 3, "dt": .02, "noise_std": .05,
        "residual_reward": True, "train_episodes": 3, "steps": 4, "epochs": 2, "batch_size": 2,
        "learning_rate": .001, "gradient_clip": 10., "rollout_horizon": 2, "rollout_weight": .5,
        "reward_scale": 4., "kl_weight": .01, "kl_balance": .8, "free_nats": 1.,
        "optimizer": copy.deepcopy(audit.ADAM), "dtype": "torch.float32", "device": "cpu",
        "objective": "unchanged_sequence_loss", "run_status_authority": "enclosing protocol and execution receipts",
        "adapter_version": "reacher-two-observation-training-v1", "model_class": "TwoObservationHistoryGRUWorldModel",
        "mlp_width_usage": "unused; retained original paired settings metadata"}
    shapes = {
        "observation_update.weight_ih": (6, 8), "observation_update.weight_hh": (6, 2),
        "observation_update.bias_ih": (6,), "observation_update.bias_hh": (6,),
        "transition.weight_ih": (6, 6), "transition.weight_hh": (6, 2),
        "transition.bias_ih": (6,), "transition.bias_hh": (6,),
        "observation_head.0.weight": (2, 2), "observation_head.0.bias": (2,),
        "observation_head.2.weight": (4, 2), "observation_head.2.bias": (4,),
        "reward_head.0.weight": (2, 4), "reward_head.0.bias": (2,),
        "reward_head.2.weight": (1, 2), "reward_head.2.bias": (1,),
    }
    initial = {name: torch.full(shape, .125, dtype=torch.float32) for name, shape in shapes.items()}
    final = {name: value + .25 for name, value in initial.items()}
    generator = torch.Generator(device="cpu").manual_seed(410)
    states = [generator.get_state().clone()]
    orders = []
    for _ in range(2):
        orders.append(torch.randperm(3, generator=generator))
        states.append(generator.get_state().clone())
    order = seal({"version": "reacher-cache-training-engineering-v1", "seed": 410,
        "role": "engineering/training-audit/orders", "orders": torch.stack(orders),
        "initial_rng": states[0], "final_rng": states[-1]})
    packet = torch.zeros(3, 5, 8)
    packet[..., :2] = 1
    packet[..., 4:6] = torch.tensor([.1, -.1])
    packet[..., 6] = 1
    for case, absent in ((0, [2, 3]), (2, [1])):
        for age, step in enumerate(absent, 1):
            packet[case, step, :4] = 0
            packet[case, step, 6] = 0
            packet[case, step, 7] = age * .02
    data = {"packets": packet, "commands": torch.zeros(3, 4, 2), "rewards": torch.zeros(3, 4)}
    initial_digest, data_digest = audit.tensor_hash(initial), audit.tensor_hash(data)
    provenance, sources, runtime = {"synthetic": "manual tensors, no training"}, {"synthetic.py": "a" * 64}, {"torch": "synthetic"}
    group = {key: copy.deepcopy(value) for key, value in audit.ADAM.items() if key != "name"}
    group.update(lr=.001, betas=(.9, .999), params=list(range(16)))
    optimizer = {"param_groups": [group], "state": {
        index: {"step": torch.tensor(4.), "exp_avg": torch.full(shape, .01), "exp_avg_sq": torch.full(shape, .001)}
        for index, shape in enumerate(shapes.values())}}
    logs = []
    for update in range(4):
        epoch, batch = divmod(update, 2)
        indices = order["orders"][epoch, batch * 2:(batch + 1) * 2]
        actual = packet[indices]
        work = audit.expected_work(settings, len(indices))
        logs.append({"kind": "two_observation_gru", "epoch": epoch, "batch": batch,
            "update": update + 1, "optimizer_steps": update + 1, "indices": indices.tolist(),
            "indices_sha256": audit.tensor_hash({"indices": indices}),
            "anchor": {"loss": 1.625, "observation_mse": .125, "reward_mse": .25, "kl_nats": 0.,
                "rollout_observation_mse": .5, "rollout_reward_mse": .125,
                "valid_observation_targets": float(actual[:, 1:, 6].sum()),
                "valid_rollout_starts": float(actual[:, :3, 6].sum())},
            "loss": 1.625, "gradient_norm": 10. if update < 3 else 11., "gradient_clipped": update == 3,
            "work": work, "costs": {"forward_seconds": .01, "backward_seconds": .02,
                "gradient_clip_seconds": .005, "optimizer_seconds": .005, "batch_wall_seconds": .05},
            "observed_neural_sample_calls": {key: work["operations"][key] for key in audit.KERNEL_FIELDS},
            "rng_identity_sha256": audit.state_hash(audit.rng_identity(order, states, update + 1, 2))})
    checkpoint = {"version": "reacher-two-observation-training-v1", "kind": "two_observation_gru",
        "settings": copy.deepcopy(settings), "model_configuration": audit.model_configuration(settings),
        "initialization": {"weights": copy.deepcopy(initial), "tensor_sha256": initial_digest},
        "orders": copy.deepcopy(order), "data_sha256": data_digest, "provenance": copy.deepcopy(provenance),
        "source_sha256": copy.deepcopy(sources), "runtime": copy.deepcopy(runtime),
        "student_state": copy.deepcopy(final), "optimizer_state": optimizer,
        "optimizer_group_names": [list(shapes)], "successful_updates": 4, "optimizer_steps": 4,
        "cursor": {"epoch": 2, "batch": 0}, "rng": audit.rng_identity(order, states, 4, 2),
        "failed": False, "failure": None,
        "last_attempt": {"status": "completed", "cursor": {"epoch": 1, "batch": 1},
            "completed_neural_sample_calls": copy.deepcopy(logs[-1]["observed_neural_sample_calls"]), "wall_seconds": .05},
        "setup_wall_seconds": .01, "training_wall_seconds": .24, "restoration_wall_seconds": 0.}
    case = {"checkpoint": checkpoint, "logs": logs, "initial_weights": initial, "orders": order, "public_data": data,
        "settings": settings, "expected_initial_sha256": initial_digest,
        "expected_orders_sha256": audit.state_hash(order), "expected_data_sha256": data_digest,
        "provenance": provenance, "source_sha256": sources, "runtime": runtime, "final_weights": final}
    rechain(case)
    return case


def test_complete_manual_payload_partial_batches_and_padded_work_without_execution(monkeypatch):
    case = fixture()
    before = torch.get_rng_state().clone()
    forbid = lambda *a, **k: (_ for _ in ()).throw(AssertionError("No model/optimizer/ambient RNG"))
    monkeypatch.setattr(torch.nn.GRUCell, "__init__", forbid)
    monkeypatch.setattr(torch.optim.Adam, "__init__", forbid)
    monkeypatch.setattr(torch, "manual_seed", forbid)
    result = audit.audit_training(**case)
    assert torch.equal(before, torch.get_rng_state())
    assert result["updates"] == result["optimizer_steps"] == result["logged_updates"] == 4
    assert result["parameters"] == 163
    assert result["historical_order_permutations_checked"] == 2
    assert result["historical_rng_states_checked"] == 3
    assert result["work"] == {"public_assimilate_samples": 24, "advance_samples": 60,
        "replayed_observation_update_samples": 288, "replayed_transition_samples": 264,
        "gru_cell_sample_calls": 612, "linear_layer_sample_calls": 1296,
        "analytic_reward_sample_calls": 324, "dense_affine_macs": 39960}
    assert result["phase_seconds"]["batch_wall_seconds"] == .2
    assert result["deployment_weights_compared"] is True
    assert result["new_model_calls"] == result["new_optimizer_steps"] == result["new_native_calls"] == 0
    assert result["new_scientific_streams"] == 0
    json.dumps(result, allow_nan=False)


def test_hash_format_has_independent_minimal_byte_and_type_witnesses():
    metadata = b'["x","torch.float32",[2]]'
    raw = struct.pack("<ff", 1., -2.)
    payload = b"OpenJev named tensors v1\0" + struct.pack(">Q", len(metadata)) + metadata + struct.pack(">Q", len(raw)) + raw
    assert audit.tensor_hash({"x": torch.tensor([1., -2.])}) == hashlib.sha256(payload).hexdigest()
    canonical = b'["dict",[["str","a",["int",1]],["str","b",["tuple",[["bool",true],["NoneType",null]]]]]]'
    assert audit.state_hash({"b": (True, None), "a": 1}) == hashlib.sha256(canonical).hexdigest()
    assert audit.state_hash({"a": 1}) != audit.state_hash({"a": True})


def test_no_input_mutation_and_optional_external_deployment_comparison():
    case = fixture()
    before = audit.state_hash(case)
    audit.audit_training(**case)
    assert audit.state_hash(case) == before
    case["final_weights"] = None
    assert audit.audit_training(**case)["deployment_weights_compared"] is False


@pytest.mark.parametrize("change", [{"rollout_weight": 0.}, {"rollout_horizon": 5}])
def test_disabled_or_too_long_rollout_still_charges_full_reconstruction(change):
    settings = fixture()["settings"]
    settings.update(change)
    work = audit.expected_work(settings, 2)
    assert work["interfaces"]["batch_forward_calls"]["student_rollout_advance"] == 0
    assert work["operations"]["advance_samples"] == 8
    assert work["operations"]["replayed_observation_update_samples"] == 96
    assert work["operations"]["replayed_transition_samples"] == 88
    assert work["operations"]["gru_cell_sample_calls"] == 192
    assert work["operations"]["linear_layer_sample_calls"] == 384


def test_negative_missing_age_not_accepted_by_float_tolerance():
    case = fixture()
    case["settings"]["dt"] = 1e-9
    packet = case["public_data"]["packets"]
    packet[packet[..., 6] == 0, 7] = -1e-9
    case["expected_data_sha256"] = audit.tensor_hash(case["public_data"])
    with pytest.raises(ValueError, match="nonnegative age"):
        audit.audit_training(**case)


def test_model_configuration_matches_constructor_float_normalization():
    case = fixture()
    case["settings"].update(dt=1, noise_std=0)
    case["checkpoint"]["settings"] = copy.deepcopy(case["settings"])
    case["public_data"]["packets"][..., 7] *= 50
    case["expected_data_sha256"] = audit.tensor_hash(case["public_data"])
    case["checkpoint"]["data_sha256"] = case["expected_data_sha256"]
    # Actual constructors normalize these two scalar attributes to Python float
    # even though the external dataclass configuration retains integer inputs.
    expected = case["checkpoint"]["model_configuration"]
    expected.update(dt=1.0, noise_std=0.0)
    for row in case["logs"]:
        row["work"]["operations"]["configuration"].update(dt=1.0, noise_std=0.0)
    config = audit.model_configuration(case["settings"])
    assert type(config["dt"]) is type(config["noise_std"]) is float
    assert audit.state_hash(config) == audit.state_hash(expected)
    rechain(case)
    assert audit.audit_training(**case)["updates"] == 4


def test_clip_flag_uses_float32_scalar_threshold_comparison():
    case = fixture()
    threshold = math.nextafter(10., 0.)
    assert 10. > threshold and float(torch.tensor(threshold, dtype=torch.float32)) == 10.
    case["settings"]["gradient_clip"] = threshold
    case["checkpoint"]["settings"]["gradient_clip"] = threshold
    rechain(case)
    assert audit.audit_training(**case)["updates"] == 4
    case["logs"][0]["gradient_clipped"] = True
    rechain(case)
    with pytest.raises(ValueError, match="clip flag"):
        audit.audit_training(**case)


@pytest.mark.parametrize("field", ["expected_initial_sha256", "expected_orders_sha256", "expected_data_sha256", "expected_checkpoint_sha256"])
def test_externally_pinned_hashes_cannot_be_replaced_by_self_claims(field):
    case = fixture()
    case[field] = "f" * 64
    with pytest.raises(ValueError):
        audit.audit_training(**case)


@pytest.mark.parametrize("corruption", ["missing_field", "extra_field", "kind", "configuration", "settings", "provenance", "source", "runtime",
    "failed", "failure", "missing_tensor", "tensor_shape", "tensor_dtype", "nan_tensor", "deployment", "initial_payload",
    "order_payload", "updates", "bool_updates", "optimizer_steps", "cursor", "rng_current", "rng_consumed", "parameter_names",
    "adam_ids", "adam_recipe", "adam_slot_missing", "adam_slot_extra", "adam_step", "adam_shape", "adam_dtype", "adam_nan", "adam_negative",
    "last_cursor", "last_kernels", "last_status", "last_seconds", "last_extra", "training_time", "negative_time"])
def test_resealed_semantically_invalid_checkpoint_rejected(corruption):
    case = fixture()
    value = case["checkpoint"]
    key = "observation_update.weight_ih"
    if corruption == "missing_field":
        del value["provenance"]
    elif corruption == "extra_field":
        value["teacher"] = {}
    elif corruption in ("kind", "configuration", "settings", "provenance", "source", "runtime"):
        if corruption == "kind":
            value["kind"] = "residual_gru"
        elif corruption == "configuration":
            value["model_configuration"]["max_real_packets"] = 3
        elif corruption == "settings":
            value["settings"]["learning_rate"] = .01
        else:
            value[{"source": "source_sha256"}.get(corruption, corruption)] = {"changed": "b" * 64}
    elif corruption == "failed":
        value["failed"] = True
    elif corruption == "failure":
        value["failure"] = {"error": "retained"}
    elif corruption == "missing_tensor":
        del value["student_state"][key]
    elif corruption == "tensor_shape":
        value["student_state"][key] = torch.zeros(6, 7)
    elif corruption == "tensor_dtype":
        value["student_state"][key] = value["student_state"][key].double()
    elif corruption == "nan_tensor":
        value["student_state"][key][0, 0] = float("nan")
    elif corruption == "deployment":
        case["final_weights"][key][0, 0] += 1
    elif corruption == "initial_payload":
        value["initialization"]["weights"][key][0, 0] += 1
    elif corruption == "order_payload":
        value["orders"]["role"] = "different"
    elif corruption in ("updates", "bool_updates", "optimizer_steps"):
        value["optimizer_steps" if corruption == "optimizer_steps" else "successful_updates"] = True if corruption == "bool_updates" else 3
    elif corruption == "cursor":
        value["cursor"]["batch"] = 1
    elif corruption == "rng_current":
        value["rng"]["current_rng"] = torch.zeros_like(value["rng"]["current_rng"])
    elif corruption == "rng_consumed":
        value["rng"]["consumed_epoch_permutations"] = 1
    elif corruption == "parameter_names":
        value["optimizer_group_names"][0].reverse()
    elif corruption == "adam_ids":
        value["optimizer_state"]["param_groups"][0]["params"].reverse()
    elif corruption == "adam_recipe":
        value["optimizer_state"]["param_groups"][0]["weight_decay"] = .1
    elif corruption.startswith("adam_"):
        state = value["optimizer_state"]["state"]
        slot = state[0]
        if corruption == "adam_slot_missing":
            del state[0]
        elif corruption == "adam_slot_extra":
            slot["extra"] = torch.tensor(0.)
        elif corruption == "adam_step":
            slot["step"] = torch.tensor(3.)
        elif corruption == "adam_shape":
            slot["exp_avg"] = torch.zeros(1)
        elif corruption == "adam_dtype":
            slot["exp_avg"] = slot["exp_avg"].double()
        elif corruption == "adam_nan":
            slot["exp_avg"][0, 0] = float("nan")
        else:
            slot["exp_avg_sq"][0, 0] = -.1
    elif corruption.startswith("last_"):
        last = value["last_attempt"]
        if corruption == "last_cursor":
            last["cursor"]["batch"] = 0
        elif corruption == "last_kernels":
            last["completed_neural_sample_calls"]["gru_cell_sample_calls"] -= 1
        elif corruption == "last_status":
            last["status"] = "failed"
        elif corruption == "last_seconds":
            last["wall_seconds"] = .06
        else:
            last["ignored"] = True
    elif corruption == "training_time":
        value["training_wall_seconds"] = .19
    else:
        value["setup_wall_seconds"] = -1.
    # Nonfinite data is rejected during safe hashing; other cases are re-sealed
    # and externally rebound so semantic checks must still reject them.
    with pytest.raises(ValueError):
        seal(value)
        case["expected_checkpoint_sha256"] = value["integrity_sha256"]
        audit.audit_training(**case)


@pytest.mark.parametrize("corruption", ["missing_log", "extra_log", "extra_field", "indices", "index_hash", "epoch", "bool_update",
    "wrong_kind", "rng_hash", "loss", "mask", "rollout_mask", "kl", "clip", "work", "observed", "negative_norm",
    "cost_phase", "cost_nested", "previous_chain", "final_chain"])
def test_rechained_invalid_update_evidence_rejected(corruption):
    case = fixture()
    row = case["logs"][1]
    if corruption == "missing_log":
        case["logs"].pop()
    elif corruption == "extra_log":
        case["logs"].append(copy.deepcopy(row))
    elif corruption == "extra_field":
        row["unobserved"] = True
    elif corruption == "indices":
        row["indices"][0] = (row["indices"][0] + 1) % 3
    elif corruption == "index_hash":
        row["indices_sha256"] = "b" * 64
    elif corruption == "epoch":
        row["epoch"] = 1
    elif corruption == "bool_update":
        case["logs"][0]["update"] = True
    elif corruption == "wrong_kind":
        row["kind"] = "residual_gru"
    elif corruption == "rng_hash":
        row["rng_identity_sha256"] = "b" * 64
    elif corruption == "loss":
        row["loss"] = row["anchor"]["loss"] = 1.5
    elif corruption in ("mask", "rollout_mask", "kl"):
        row["anchor"][{"mask": "valid_observation_targets", "rollout_mask": "valid_rollout_starts", "kl": "kl_nats"}[corruption]] += 1
    elif corruption == "clip":
        row["gradient_clipped"] = True
    elif corruption == "work":
        row["work"]["operations"]["replayed_transition_samples"] -= 1
    elif corruption == "observed":
        row["observed_neural_sample_calls"]["linear_layer_sample_calls"] -= 1
    elif corruption == "negative_norm":
        row["gradient_norm"] = -1.
    elif corruption == "cost_phase":
        row["costs"]["forward_seconds"] = 0.
    elif corruption == "cost_nested":
        row["costs"]["forward_seconds"] = .1
    rechain(case)
    if corruption == "previous_chain":
        row["previous_log_sha256"] = "f" * 64
    elif corruption == "final_chain":
        case["checkpoint"]["log_chain_sha256"] = "f" * 64
        seal(case["checkpoint"])
        case["expected_checkpoint_sha256"] = case["checkpoint"]["integrity_sha256"]
    with pytest.raises(ValueError):
        audit.audit_training(**case)


@pytest.mark.parametrize("corruption", ["orders_permutation", "orders_seed", "orders_initial_rng", "orders_final_rng",
    "data_mask", "data_age", "data_target", "data_commands", "data_first_valid", "data_dtype", "data_extra"])
def test_rebound_orders_and_public_tensors_still_require_actual_semantics(corruption):
    case = fixture()
    if corruption.startswith("orders_"):
        orders = case["orders"]
        if corruption == "orders_permutation":
            orders["orders"][0] = orders["orders"][0].roll(1)
        elif corruption == "orders_seed":
            orders["seed"] = True  # Invalid type: no additional seed is consumed.
        else:
            orders[corruption.removeprefix("orders_")].zero_()
        seal(orders)
        case["expected_orders_sha256"] = audit.state_hash(orders)
    else:
        data = case["public_data"]
        if corruption == "data_mask":
            data["packets"][0, 2, 0] = 1.
        elif corruption == "data_age":
            data["packets"][0, 2, 7] = 0.
        elif corruption == "data_target":
            data["packets"][0, 3, 4] += .01
        elif corruption == "data_commands":
            data["commands"][0, 0, 0] = 2.
        elif corruption == "data_first_valid":
            data["packets"][0, 0, :4] = 0
            data["packets"][0, 0, 6] = 0
        elif corruption == "data_dtype":
            data["rewards"] = data["rewards"].double()
        else:
            data["velocity"] = torch.zeros(3, 5, 2)
        case["expected_data_sha256"] = audit.tensor_hash(data)
    with pytest.raises(ValueError):
        audit.audit_training(**case)


def test_whole_order_hash_is_not_inner_integrity_and_cap_preserves_ambient_rng():
    case = fixture()
    assert case["expected_orders_sha256"] != case["orders"]["integrity_sha256"]
    case["expected_orders_sha256"] = case["orders"]["integrity_sha256"]
    with pytest.raises(ValueError, match="whole sealed orders"):
        audit.audit_training(**case)
    case = fixture()
    before = torch.get_rng_state().clone()
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("synthetic historical-order validation cap")
    with pytest.raises(TimeoutError):
        audit.audit_training(**case, deadline_check=stop)
    assert torch.equal(before, torch.get_rng_state())


def test_dependency_boundary_has_no_model_trainer_optimizer_or_io_import():
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert set(modules) <= {"__future__", "hashlib", "json", "math", "struct", "collections.abc", "torch"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"open", "eval", "exec", "__import__"} for node in ast.walk(tree))
    assert "torch.optim" not in source and "torch.nn" not in source and "torch.load" not in source
