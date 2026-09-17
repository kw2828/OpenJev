"""Architecture invariants only: no optimizer steps or benchmark evaluation."""

import copy
import math

import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_candidate import (
    ARCHITECTURE,
    ARMS,
    CHECKPOINT_VERSION,
    CandidateChess,
    encode_batch,
)
from openjev.research.chess_spatial import action_planes, encode_board


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def boards():
    return [chess.Board(), chess.Board("7k/8/8/8/8/8/p7/7K b - - 7 19")]


def call_parent(model, batch, depth=None):
    return model(batch["observations"], batch["candidates"], batch["legal_mask"], depth=depth)


def test_common_initialization_rng_and_stored_active_parameters():
    torch.manual_seed(902)
    before = torch.random.get_rng_state().clone()
    parent = AnchorChess("residual", seed=17)
    reference = CandidateChess("direct", seed=17)
    for arm in ARMS:
        model = CandidateChess(arm, seed=17)
        for name, parameter in parent.state_dict().items():
            assert torch.equal(model.state_dict()[name], parameter), name
        for name, parameter in reference.state_dict().items():
            assert torch.equal(model.state_dict()[name], parameter), (arm, name)
        counts = model.parameter_counts()
        assert counts["stored"] == 43854
        assert counts["active"] == (33185 if arm == "direct" else 33313)
        assert counts["inactive_auxiliary"] == 10541
        assert counts["inactive_action_projection"] == (128 if arm == "direct" else 0)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("seed", [17, 29])
@pytest.mark.parametrize("depth", [1, 4, 8])
def test_direct_logits_values_hidden_and_gradients_match_frozen_parent_exactly(seed, depth):
    batch, _ = encode_batch(boards(), "direct")
    parent = AnchorChess("residual", seed=seed, width=8)
    direct = CandidateChess("direct", seed=seed, width=8)
    expected = call_parent(parent, batch, depth)
    actual = direct(**batch, depth=depth)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for logits, value, _ in (expected, actual):
        (F.cross_entropy(logits, torch.zeros(2, dtype=torch.long)) + 0.5 * value.square().mean()).backward()
    for name, parameter in parent.named_parameters():
        other = dict(direct.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    assert direct.action_projection.weight.grad is None


@pytest.mark.parametrize("arm", ARMS)
def test_root_value_is_computed_once_and_identical_across_arms(arm):
    batch, _ = encode_batch(boards(), arm)
    model = CandidateChess(arm, seed=19, width=8)
    reference = AnchorChess("residual", seed=19, width=8)
    expected = call_parent(reference, batch)
    value_calls = []
    hook = model.value_head.register_forward_hook(
        lambda _, inputs, output: value_calls.append((inputs[0].shape, output.shape))
    )
    logits, value, root = model(**batch, candidate_chunk_size=3)
    hook.remove()
    assert value_calls == [(torch.Size([2, 8]), torch.Size([2, 1]))]
    torch.testing.assert_close(root, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(value, expected[1], rtol=0, atol=0)
    assert logits.shape == batch["legal_mask"].shape
    value.square().sum().backward()
    expected[1].square().sum().backward()
    for name, parameter in reference.named_parameters():
        actual = dict(model.named_parameters())[name]
        if parameter.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, parameter.grad, rtol=0, atol=0)
    assert model.action_projection.weight.grad is None


@pytest.mark.parametrize("arm", ["action_only", "delta", "full_afterstate"])
def test_candidate_branch_matches_manual_shared_core_and_linear_delta(arm):
    batch, _ = encode_batch(boards(), arm)
    model = CandidateChess(arm, seed=31, width=4, root_depth=2, branch_depth=2)
    actual, value, root = model(**batch, candidate_chunk_size=1000)
    indices = batch["legal_mask"].nonzero()
    actions = batch["candidates"][batch["legal_mask"]]
    projected = model.action_projection(action_planes(actions))
    hidden = root[indices[:, 0]] + projected
    if arm == "delta":
        delta = batch["successors"] - batch["observations"][indices[:, 0]]
        hidden = hidden + F.conv2d(delta, model.encoder[0].weight, bias=None, padding=1)
        # The encoder bias must cancel from a linear input correction.
        left = batch["successors"].double()
        right = batch["observations"][indices[:, 0]].double()
        weight, bias = model.encoder[0].weight.double(), model.encoder[0].bias.double()
        correction = F.conv2d(left, weight, bias, padding=1) - F.conv2d(right, weight, bias, padding=1)
        torch.testing.assert_close(
            correction, F.conv2d(left - right, weight, padding=1), atol=1e-14, rtol=1e-12
        )
    elif arm == "full_afterstate":
        hidden = model.encoder(batch["successors"])
        for _ in range(model.depth):
            hidden = model.core(hidden)
        hidden = hidden + projected
    for _ in range(model.branch_depth):
        hidden = model.core(hidden)
    expected = model._candidate_scores(hidden, actions)
    torch.testing.assert_close(actual[batch["legal_mask"]], expected, atol=0, rtol=0)
    assert value.shape == (2,)


def test_no_native_successors_are_constructed_for_direct_or_action_only(monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("Native successor construction is forbidden for this control")

    monkeypatch.setattr(chess.Board, "push_uci", forbidden)
    for arm in ("direct", "action_only"):
        batch, menus = encode_batch(boards(), arm)
        assert "successors" not in batch
        assert len(menus) == 2


@pytest.mark.parametrize("arm", ["delta", "full_afterstate"])
def test_encoding_retains_board_history_and_ignores_attached_labels(arm):
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "e2e4"):
        board.push_uci(uci)
    before = board.fen(en_passant="fen"), list(board.move_stack)
    first, menus = encode_batch([board], arm)
    board.teacher_target = "g7g5"
    board.score_cp = 9999
    second, changed_menus = encode_batch([board], arm)
    assert changed_menus == menus and first.keys() == second.keys()
    assert set(first) == {"observations", "candidates", "legal_mask", "successors"}
    for key in first:
        assert torch.equal(first[key], second[key])
    assert before == (board.fen(en_passant="fen"), list(board.move_stack))
    assert first["successors"].shape[0] == int(first["legal_mask"].sum())


def test_black_mirror_root_orientation_and_fullmove_counter_are_explicit():
    board = chess.Board()
    board.push_uci("e2e4")
    mirrored = board.mirror()
    batch, menus = encode_batch([board, mirrored], "delta")
    torch.testing.assert_close(batch["observations"][0], batch["observations"][1], atol=0, rtol=0)
    offset = len(menus[0])
    for index, uci in enumerate(menus[0]):
        move = chess.Move.from_uci(uci)
        mirrored_uci = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        ).uci()
        other = menus[1].index(mirrored_uci)
        torch.testing.assert_close(
            batch["candidates"][0, index], batch["candidates"][1, other], atol=0, rtol=0
        )
        left, right = batch["successors"][index], batch["successors"][offset + other]
        torch.testing.assert_close(left[:18], right[:18], atol=0, rtol=0)
        # Native fullmove increments after black, but not the mirrored white move.
        torch.testing.assert_close(left[18], torch.full_like(left[18], math.log1p(2) / 10), rtol=0, atol=0)
        torch.testing.assert_close(right[18], torch.full_like(right[18], math.log1p(1) / 10), rtol=0, atol=0)
    black_child = board.copy(stack=True)
    black_child.push_uci("g7g5")
    index = menus[0].index("g7g5")
    np.testing.assert_array_equal(
        batch["successors"][index].numpy(), encode_board(black_child, perspective=chess.BLACK)
    )
    assert not np.array_equal(batch["successors"][index].numpy(), encode_board(black_child))


@pytest.mark.parametrize(
    "fen,uci",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 12 20", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 20", "e8c8"),
        ("7k/8/8/3pP3/8/8/8/7K w - d6 0 12", "e5d6"),
        ("7k/8/8/8/3Pp3/8/8/7K b - d3 0 12", "e4d3"),
        ("7k/P7/8/8/8/8/8/7K w - - 7 19", "a7a8q"),
        ("7k/P7/8/8/8/8/8/7K w - - 7 19", "a7a8n"),
        ("7k/8/8/8/8/8/p7/7K b - - 7 19", "a2a1r"),
    ],
)
def test_special_moves_encode_all_native_piece_rights_ep_and_counter_changes(fen, uci):
    board = chess.Board(fen)
    batch, menus = encode_batch([board], "delta")
    index = menus[0].index(uci)
    child = board.copy(stack=True)
    child.push_uci(uci)
    expected = encode_board(child, perspective=board.turn)
    np.testing.assert_array_equal(batch["successors"][index].numpy(), expected)
    delta = batch["successors"][index] - batch["observations"][0]
    np.testing.assert_array_equal(delta.numpy(), expected - encode_board(board))
    if board.is_castling(chess.Move.from_uci(uci)):
        assert torch.count_nonzero(delta[:12]) == 4  # king and rook source/destination
        assert torch.count_nonzero(delta[12:14]) == 128  # both own rights removed
    if board.is_en_passant(chess.Move.from_uci(uci)):
        assert torch.count_nonzero(delta[:12]) == 3
        assert torch.count_nonzero(delta[16]) == 1 and child.ep_square is None
    if chess.Move.from_uci(uci).promotion:
        assert torch.count_nonzero(delta[:12]) == 2
        assert torch.count_nonzero(delta[17]) == 64
    if board.turn == chess.BLACK:
        assert torch.count_nonzero(delta[18]) == 64
    else:
        assert not torch.count_nonzero(delta[18])


def test_uncapturable_ep_square_is_preserved_and_clears_on_quiet_reply():
    board = chess.Board()
    batch, menus = encode_batch([board], "delta")
    successor = batch["successors"][menus[0].index("e2e4")]
    assert successor[16, 2, 4] == 1 and successor[16].sum() == 1
    board.push_uci("e2e4")
    assert not board.has_legal_en_passant() and board.ep_square == chess.E3
    reply, names = encode_batch([board], "delta")
    assert reply["observations"][0, 16].sum() == 1
    assert reply["successors"][names[0].index("g8f6"), 16].sum() == 0


@pytest.mark.parametrize("arm", ["delta", "full_afterstate"])
def test_permuting_only_successor_consequences_cannot_change_root_value(arm):
    batch, _ = encode_batch([chess.Board()], arm)
    model = CandidateChess(arm, seed=17, width=8)
    original = model(**batch)
    changed = model(**{**batch, "successors": batch["successors"].flip(0)})
    torch.testing.assert_close(original[1], changed[1], rtol=0, atol=0)
    torch.testing.assert_close(original[2], changed[2], rtol=0, atol=0)
    assert not torch.equal(original[0], changed[0])


@pytest.mark.parametrize("arm", ARMS)
def test_candidate_permutation_padding_chunking_and_input_preservation(arm):
    batch, _ = encode_batch(boards(), arm)
    original = {key: value.clone() for key, value in batch.items()}
    model = CandidateChess(arm, seed=23, width=8)
    logits, value, hidden = model(**batch, candidate_chunk_size=1000)
    permutation = torch.arange(batch["candidates"].shape[1] - 1, -1, -1)
    changed = {
        **batch,
        "candidates": batch["candidates"][:, permutation],
        "legal_mask": batch["legal_mask"][:, permutation],
    }
    if "successors" in batch:
        source_index = torch.full(batch["legal_mask"].shape, -1, dtype=torch.long)
        source_index[batch["legal_mask"]] = torch.arange(len(batch["successors"]))
        successor_order = source_index[:, permutation][changed["legal_mask"]]
        changed["successors"] = batch["successors"][successor_order]
    permuted = model(**changed, candidate_chunk_size=3)
    torch.testing.assert_close(permuted[0], logits[:, permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(permuted[1], value, atol=0, rtol=0)
    torch.testing.assert_close(permuted[2], hidden, atol=0, rtol=0)
    assert torch.isneginf(logits[~batch["legal_mask"]]).all()
    assert torch.equal(
        logits.softmax(-1)[~batch["legal_mask"]], torch.zeros_like(logits[~batch["legal_mask"]])
    )
    for key in batch:
        assert torch.equal(batch[key], original[key])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_active_branch_gradients_and_recurrence_counts_without_updates(arm, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = CandidateChess(arm, seed=11, width=8).to(device)
    batch, menus = encode_batch(boards(), arm, device=device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    core_calls = []
    handle = model.core.conv1.register_forward_pre_hook(lambda _, inputs: core_calls.append(len(inputs[0])))
    logits, values, _ = model(**batch, candidate_chunk_size=3)
    handle.remove()
    loss = F.cross_entropy(logits, torch.zeros(2, device=device, dtype=torch.long))
    loss = loss + 0.5 * F.mse_loss(values, torch.tensor([0.3, -0.4], device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(core_calls) == sum(model.recurrence_cost(len(menu))["total_core_iterations"] for menu in menus)
    for name, parameter in model.named_parameters():
        inactive = name.startswith("aux_head.") or (arm == "direct" and name.startswith("action_projection."))
        if inactive:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        assert torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("arm", ARMS)
def test_choose_is_stateless_legal_and_reports_actual_costs(arm):
    model = CandidateChess(arm, seed=7, width=8)
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    before = board.fen(en_passant="fen"), list(board.move_stack)
    first = model.choose(board, depth=4, candidate_chunk_size=3)
    model.choose(boards()[1], depth=2)
    repeated = model.choose(chess.Board(before[0]), depth=4, candidate_chunk_size=3)
    for key in ("choice", "probabilities", "value", "arm", "seed", "depth"):
        assert first[key] == repeated[key]
    assert set(first["probabilities"]) == {m.uci() for m in board.legal_moves}
    assert sum(first["probabilities"].values()) == pytest.approx(1)
    assert first["candidate_evaluations"] == 20
    assert first["native_successors"] == (20 if arm in ("delta", "full_afterstate") else 0)
    assert (
        first["total_core_iterations"]
        == {"direct": 4, "action_only": 44, "delta": 44, "full_afterstate": 124}[arm]
    )
    assert first["probability_semantics"] == "uncalibrated legal-move softmax"
    assert first["latency_ms"] >= 0
    assert before == (board.fen(en_passant="fen"), list(board.move_stack))


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_roundtrip_with_exact_architecture_and_plan_bindings(tmp_path, arm):
    model = CandidateChess(arm, seed=29, width=8, root_depth=2, branch_depth=3)
    path = tmp_path / "weights.pt"
    model.save(path, plan_sha256="a" * 64)
    loaded = CandidateChess.load(
        path,
        expected_plan_sha256="a" * 64,
        expected_arm=arm,
        expected_seed=29,
        expected_width=8,
        expected_root_depth=2,
        expected_branch_depth=3,
    )
    assert not loaded.training
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())
    with pytest.raises(FileExistsError):
        model.save(path, plan_sha256="a" * 64)
    for expected in (
        {"expected_plan_sha256": "b" * 64},
        {"expected_arm": next(a for a in ARMS if a != arm)},
        {"expected_seed": 17},
        {"expected_width": 32},
        {"expected_root_depth": 4},
        {"expected_branch_depth": 2},
        {"expected_seed": True},
    ):
        with pytest.raises(ValueError):
            CandidateChess.load(path, **expected)
    with pytest.raises(ValueError):
        AnchorChess.load(path)


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "architecture",
        "recurrence",
        "arm",
        "extra",
        "missing",
        "shape",
        "dtype",
        "nan",
        "root_depth",
        "branch_depth",
    ],
)
def test_checkpoint_rejects_corrupt_metadata_and_tensors(tmp_path, fault):
    path = tmp_path / "good.pt"
    CandidateChess("delta", width=4).save(path, plan_sha256="a" * 64)
    value = torch.load(path, weights_only=True)
    assert value["format_version"] == CHECKPOINT_VERSION and value["architecture"] == ARCHITECTURE
    key = next(iter(value["state_dict"]))
    if fault == "format":
        value["format_version"] = "openjev-chess-anchor-v1"
    elif fault == "architecture":
        value["architecture"] = "different"
    elif fault == "recurrence":
        value["recurrence"] = "anchor"
    elif fault == "arm":
        value["arm"] = "new_arm"
    elif fault == "extra":
        value["unrecorded_option"] = 1
    elif fault == "missing":
        value["state_dict"].pop(key)
    elif fault == "shape":
        value["state_dict"][key] = value["state_dict"][key][:-1]
    elif fault == "dtype":
        value["state_dict"][key] = value["state_dict"][key].double()
    elif fault == "nan":
        value["state_dict"][key].flatten()[0] = torch.nan
    else:
        value[fault] = 0
    corrupted = tmp_path / "corrupt.pt"
    torch.save(value, corrupted)
    with pytest.raises(ValueError):
        CandidateChess.load(corrupted)


def test_parent_checkpoint_is_rejected_and_loader_uses_weights_only(tmp_path, monkeypatch):
    path = tmp_path / "parent.pt"
    AnchorChess("residual", width=4).save(path, plan_sha256="a" * 64)
    original_load = torch.load
    calls = []

    def observed_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    with pytest.raises(ValueError):
        CandidateChess.load(path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("arm", ARMS)
def test_invalid_batch_contracts_are_rejected(arm):
    model = CandidateChess(arm, width=4)
    batch, _ = encode_batch(boards(), arm)
    broken = copy.deepcopy(batch)
    broken["legal_mask"][0] = False
    with pytest.raises(ValueError):
        model(**broken)
    broken = copy.deepcopy(batch)
    broken["observations"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        model(**broken)
    if arm in ("delta", "full_afterstate"):
        broken = {**batch, "successors": batch["successors"][:-1]}
    else:
        broken = {**batch, "successors": torch.zeros(1, 19, 8, 8)}
    with pytest.raises(ValueError):
        model(**broken)
    with pytest.raises(ValueError):
        model(**batch, candidate_chunk_size=0)
    with pytest.raises(ValueError):
        encode_batch([], arm)
    with pytest.raises(ValueError):
        encode_batch([chess.Board("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1")], arm)
