"""Handwritten NumPy evidence, with no model-generated or native fixtures."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from openjev.research import reacher_two_observation_audit as audit


@pytest.fixture
def identity():
    keys = sorted(["hidden", "packet", "real_packets", "real_actions", "real_present", "real_indices",
                   "real_index", "real_target", "pending_action", "imagined_depth"])
    config = {
        "version": "two-valid-observation-history-gru-v1", "model_class": "TwoObservationHistoryGRUWorldModel",
        "hidden_size": 3, "dt": .02, "noise_std": .05, "residual_reward": True,
        "valid_observations": 2, "max_real_packets": 12, "max_issued_commands": 11,
        "episode_steps": 50, "age_rtol": 1e-5, "age_atol": 1e-6,
        "anchor": "older of last two actual valid observations; first observation before second exists",
        "real_history": "sanitized actual public packets and issued commands; integer indices; left padding",
        "assimilation": "zero-start reconstruction at every real boundary; initial-or-one-selected-advance phase",
        "replay": "all twelve assimilations and eleven full transitions, including padding, without detach",
        "imagined_rollout": "private recurrent hidden/packet; never appends real evidence",
        "overflow": "reject before dropping required anchor or command",
        "state_keys": keys, "run_status_authority": "enclosing protocol and execution receipts",
    }
    return {"version": "reacher-two-observation-control-v1", "kind": "two_observation_gru",
            "model_class": "TwoObservationHistoryGRUWorldModel", "configuration": config, "width": 3,
            "weight_tensor_sha256": "a" * 64, "parameter_count": 266, "parameter_tensor_bytes": 1064}


def public_prefix(visible):
    """Deterministic handwritten geometry; no random generator or true state."""
    visible = np.asarray(visible, dtype=bool)
    if visible.ndim == 1:
        visible = visible[None]
    b, t = visible.shape
    packets = np.zeros((b, t, 8), dtype=np.float32)
    for case in range(b):
        last = 0
        for step in range(t):
            if visible[case, step]:
                last = step
                packets[case, step, :4] = [1, .8, step / 100, case / 10]
            packets[case, step, 4:] = [.1 + case / 100, -.12, int(visible[case, step]), (step - last) * .02]
    commands = np.empty((b, t - 1, 2), dtype=np.float32)
    for case in range(b):
        for step in range(t - 1):
            commands[case, step] = [step / 100, -.2 - case / 100]
    return packets, commands


def saved_pair(packets, commands, *, anchors, hidden_size=3):
    """Build explicitly chosen suffixes, without calling the audited function."""
    b, t = packets.shape[:2]
    root = {"hidden": np.full((b, hidden_size), .25, dtype=np.float32),
            "packet": packets[:, -1].copy(), "real_packets": np.zeros((b, 12, 8), dtype=np.float32),
            "real_actions": np.zeros((b, 11, 2), dtype=np.float32), "real_present": np.zeros((b, 12), dtype=bool),
            "real_indices": np.full((b, 12), -1, dtype=np.int64),
            "real_index": np.full((b, 1), t - 1, dtype=np.int64), "real_target": packets[:, 0, 4:6].copy(),
            "pending_action": np.zeros((b, 2), dtype=np.float32), "imagined_depth": np.zeros((b, 1), dtype=np.int64)}
    for case, anchor in enumerate(anchors):
        start = 12 - (t - anchor)
        root["real_packets"][case, start:] = packets[case, anchor:]
        root["real_present"][case, start:] = True
        root["real_indices"][case, start:] = list(range(anchor, t))
        if t - anchor > 1:
            root["real_actions"][case, start:] = commands[case, anchor:]
    carried = copy.deepcopy(root)
    action = np.tile(np.array([[.125, -.5]], dtype=np.float32), (b, 1))
    carried["pending_action"] = action.copy()
    carried["imagined_depth"][:] = 1
    carried["hidden"][:] = -.75  # Arbitrary learned output, deliberately not independently verified.
    carried["packet"][:, :4] = [123, -45, 67, -89]
    carried["packet"][:, 6] = 0
    carried["packet"][:, 7] += np.float32(.02)
    return root, carried, action


def check(identity, packets, commands, root, carried, action, *, step=None):
    return audit.audit_decision(public_packets=packets, issued_commands=commands,
        step=packets.shape[1] - 1 if step is None else step, root_state=root, carried_state=carried,
        selected_issued_command=action, recorded_identity=copy.deepcopy(identity), expected_identity=identity)


@pytest.mark.parametrize(("visible", "anchor"), [([1], 0), ([1, 0, 0], 0), ([1, 1], 0),
    ([1, 1, 1], 1), ([1, 0, 1, 0, 0], 0), ([1, 1, 0, 1, 0], 1)])
def test_explicit_anchor_startup_and_action_alignment(identity, visible, anchor):
    packets, commands = public_prefix(visible)
    root, carried, action = saved_pair(packets, commands, anchors=[anchor])
    result = check(identity, packets, commands, root, carried, action)
    assert result["anchor_indices"] == [anchor]
    assert result["retained_packets_per_case"] == [len(visible) - anchor]
    assert result["public_history_verified"] and not result["learned_hidden_numerically_verified"]
    assert not result["predicted_angles_numerically_verified"]
    assert result["new_native_calls"] == result["new_model_calls"] == 0


@pytest.mark.parametrize(("visible", "anchor"), [([1, 1] + [0] * 10, 0), ([1, 1] + [0] * 10 + [1], 1)])
def test_full_twelve_slots_and_thirteenth_scratch_reacquisition(identity, visible, anchor):
    packets, commands = public_prefix(visible)
    root, carried, action = saved_pair(packets, commands, anchors=[anchor])
    assert check(identity, packets, commands, root, carried, action)["retained_packets_per_case"] == [12]
    assert np.array_equal(root["real_actions"][0], commands[0, anchor:])


def test_case_specific_anchors_and_return_no_aliasing(identity):
    packets, commands = public_prefix([[1, 1, 0, 0, 1], [1, 0, 1, 1, 0]])
    root, carried, action = saved_pair(packets, commands, anchors=[1, 2])
    assert check(identity, packets, commands, root, carried, action)["anchor_indices"] == [1, 2]
    expected = audit.reconstruct_public_history(packets, commands, step=4, expected_configuration=identity["configuration"])
    for key in expected:
        assert np.array_equal(expected[key], root[key])
        assert not np.shares_memory(expected[key], packets)
        assert not np.shares_memory(expected[key], commands)


def test_poison_in_missing_raw_angles_is_removed_not_hidden_channel(identity):
    packets, commands = public_prefix([1, 1, 0, 0])
    root, carried, action = saved_pair(packets, commands, anchors=[0])
    packets[0, 2, :4] = [np.nan, np.inf, -np.inf, 12345]
    packets[0, 3, :4] = [-500, np.nan, 98, -np.inf]
    before = packets.copy()
    assert check(identity, packets, commands, root, carried, action)["status"] == "passed"
    assert np.array_equal(before, packets, equal_nan=True)
    root["real_packets"][0, -1, 0] = 12345
    with pytest.raises(ValueError, match="actual public prefix"):
        check(identity, packets, commands, root, carried, action)


@pytest.mark.parametrize("visible", [[1] + [0] * 12, [1] + [0] * 12 + [1, 1]])
def test_reject_historical_overflow_even_if_final_suffix_would_fit(identity, visible):
    packets, commands = public_prefix(visible)
    with pytest.raises(ValueError, match="exceeds twelve"):
        audit.reconstruct_public_history(packets, commands, step=len(visible) - 1,
                                         expected_configuration=identity["configuration"])


def test_real_terminal_boundary_and_fifty_actions(identity):
    packets, commands = public_prefix([1] * 50)
    root, carried, action = saved_pair(packets, commands, anchors=[48])
    assert check(identity, packets, commands, root, carried, action)["step"] == 49
    packets, commands = public_prefix([1] * 51)
    expected = audit.reconstruct_public_history(packets, commands, step=50, expected_configuration=identity["configuration"])
    assert expected["real_indices"][0, -2:].tolist() == [49, 50]
    with pytest.raises(ValueError, match="decision requires"):
        check(identity, packets, commands, root, carried, action)
    with pytest.raises(ValueError, match="Integer real step"):
        audit.reconstruct_public_history(packets, commands, step=51, expected_configuration=identity["configuration"])


@pytest.mark.parametrize("mutation", ["future_packet", "future_command", "short_commands", "float_step", "bool_step",
    "first_missing", "visible_poison", "target_poison", "age_poison", "wrong_age", "visible_age", "target_drift",
    "old_target_drift", "validity_half", "command_unclipped", "command_nan", "packet_float64", "commands_float64"])
def test_reject_bad_actual_prefix(identity, mutation):
    packets, commands = public_prefix([1, 1, 0, 1, 0])
    step = 4
    if mutation == "future_packet":
        packets = np.concatenate((packets, packets[:, -1:]), 1)
    elif mutation == "future_command":
        commands = np.concatenate((commands, commands[:, -1:]), 1)
    elif mutation == "short_commands":
        commands = commands[:, :-1]
    elif mutation == "float_step":
        step = 4.0
    elif mutation == "bool_step":
        step = True
    elif mutation == "first_missing":
        packets[0, 0, 6] = 0
    elif mutation == "visible_poison":
        packets[0, 0, 0] = np.nan
    elif mutation == "target_poison":
        packets[0, 2, 4] = np.inf
    elif mutation == "age_poison":
        packets[0, 2, 7] = np.nan
    elif mutation == "wrong_age":
        packets[0, 2, 7] = .04
    elif mutation == "visible_age":
        packets[0, 1, 7] = 1e-8
    elif mutation in {"target_drift", "old_target_drift"}:
        packets[0, 4 if mutation == "target_drift" else 0, 4] += .01
    elif mutation == "validity_half":
        packets[0, 2, 6] = .5
    elif mutation == "command_unclipped":
        commands[0, 0, 0] = 1.01
    elif mutation == "command_nan":
        commands[0, 0, 0] = np.nan
    elif mutation == "packet_float64":
        packets = packets.astype(np.float64)
    elif mutation == "commands_float64":
        commands = commands.astype(np.float64)
    with pytest.raises(ValueError):
        audit.reconstruct_public_history(packets, commands, step=step, expected_configuration=identity["configuration"])


@pytest.mark.parametrize("mutation", ["root_packet", "retained_future_angle", "dropped_anchor", "retained_old_angle",
    "wrong_action_alignment", "padding_action", "padding_packet", "padding_index", "wrong_clock", "root_depth",
    "root_pending", "carried_real_packet", "carried_real_action", "carried_indices", "carried_clock", "carried_target",
    "carried_valid", "carried_age", "carried_depth", "wrong_pending", "actual_command", "command_bounds",
    "extra_state", "missing_state", "hidden_shape", "hidden_nan", "prediction_nan", "wrong_float_dtype", "wrong_int_dtype"])
def test_reject_state_or_selected_command_corruption(identity, mutation):
    packets, commands = public_prefix([1, 1, 1, 0])
    root, carried, action = saved_pair(packets, commands, anchors=[1])
    if mutation == "root_packet":
        root["packet"][0, 0] = 1
    elif mutation == "retained_future_angle":
        root["real_packets"][0, -2, 0] += .1
    elif mutation == "dropped_anchor":
        root["real_present"][0, -3] = False
    elif mutation == "retained_old_angle":
        root["real_packets"][0, -4] = packets[0, 0]
        root["real_present"][0, -4] = True
        root["real_indices"][0, -4] = 0
    elif mutation == "wrong_action_alignment":
        root["real_actions"][0, -1] = commands[0, 0]
    elif mutation == "padding_action":
        root["real_actions"][0, 0, 0] = .1
    elif mutation == "padding_packet":
        root["real_packets"][0, 0, 4] = .1
    elif mutation == "padding_index":
        root["real_indices"][0, 0] = 0
    elif mutation == "wrong_clock":
        root["real_index"][:] += 1
    elif mutation == "root_depth":
        root["imagined_depth"][:] = 1
    elif mutation == "root_pending":
        root["pending_action"][:] = action
    elif mutation == "carried_real_packet":
        carried["real_packets"][0, -1, 0] = .2
    elif mutation == "carried_real_action":
        carried["real_actions"][0, -1] = action[0]
    elif mutation == "carried_indices":
        carried["real_indices"][0, -1] += 1
    elif mutation == "carried_clock":
        carried["real_index"][:] += 1
    elif mutation == "carried_target":
        carried["packet"][0, 4] += .01
    elif mutation == "carried_valid":
        carried["packet"][0, 6] = 1
    elif mutation == "carried_age":
        carried["packet"][0, 7] += .02
    elif mutation == "carried_depth":
        carried["imagined_depth"][:] = 2
    elif mutation == "wrong_pending":
        carried["pending_action"][0, 0] += .1
    elif mutation == "actual_command":
        action[0, 0] += .1
    elif mutation == "command_bounds":
        action[0, 0] = 2
        carried["pending_action"][0, 0] = 2
    elif mutation == "extra_state":
        root["future"] = np.zeros(1)
    elif mutation == "missing_state":
        del carried["real_actions"]
    elif mutation == "hidden_shape":
        root["hidden"] = np.zeros((1, 4), dtype=np.float32)
    elif mutation == "hidden_nan":
        root["hidden"][0, 0] = np.nan
    elif mutation == "prediction_nan":
        carried["packet"][0, 0] = np.inf
    elif mutation == "wrong_float_dtype":
        root["real_packets"] = root["real_packets"].astype(np.float64)
    elif mutation == "wrong_int_dtype":
        root["real_indices"] = root["real_indices"].astype(np.int32)
    with pytest.raises(ValueError):
        check(identity, packets, commands, root, carried, action)


def test_cannot_rebind_identity_without_callers_authority(identity):
    packets, commands = public_prefix([1])
    root, carried, action = saved_pair(packets, commands, anchors=[0])
    recorded = copy.deepcopy(identity)
    recorded["weight_tensor_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="external authenticated binding"):
        audit.audit_decision(public_packets=packets, issued_commands=commands, step=0, root_state=root,
            carried_state=carried, selected_issued_command=action, recorded_identity=recorded, expected_identity=identity)


@pytest.mark.parametrize(("where", "key", "value"), [("top", "model_class", "GRUResidualRewardWorldModel"),
    ("top", "version", "reacher-cache-control-v1"), ("top", "parameter_count", 267),
    ("top", "parameter_tensor_bytes", 1), ("top", "weight_tensor_sha256", "a"),
    ("top", "width", True), ("config", "dt", .01), ("config", "residual_reward", False),
    ("config", "max_real_packets", 13), ("config", "hidden_size", True), ("config", "noise_std", float("nan")),
    ("config", "valid_observations", True), ("config", "age_atol", .1)])
def test_reject_wrong_class_or_contract_even_if_both_bindings_agree(identity, where, key, value):
    packets, commands = public_prefix([1])
    root, carried, action = saved_pair(packets, commands, anchors=[0])
    target = identity if where == "top" else identity["configuration"]
    target[key] = value
    with pytest.raises(ValueError):
        check(identity, packets, commands, root, carried, action)


def test_age_tolerance_is_documented_but_does_not_relax_buffer_equality(identity):
    packets, commands = public_prefix([1, 0])
    packets[0, 1, 7] += np.float32(2e-7)
    root, carried, action = saved_pair(packets, commands, anchors=[0])
    assert check(identity, packets, commands, root, carried, action)["status"] == "passed"
    root["real_packets"][0, -1, 7] += np.float32(2e-7)
    with pytest.raises(ValueError, match="actual public prefix"):
        check(identity, packets, commands, root, carried, action)


def test_no_tensor_model_controller_simulator_imports():
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) <= {"__future__", "hashlib", "json", "math", "re", "collections.abc", "numpy"}
    assert "import_module" not in source and "__import__" not in source
