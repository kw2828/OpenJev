"""Synthetic arena checks: no real checkpoints, teacher or scored games."""

import copy
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch

from openjev.research import chess_capacity_arena as arena

PLAN_HASH = "a" * 64


def response(board, spec, choice=None):
    legal = sorted(m.uci() for m in board.legal_moves)
    choice = legal[0] if choice is None else choice
    side = "white" if board.turn else "black"
    return {
        "choice": choice,
        "probabilities": {uci: float(uci == choice) for uci in legal},
        "width": spec[f"{side}_width"],
        "seed": spec["seed"],
        "depth": 4,
        "mode": "recurrent",
        "recurrence": "residual",
    }


def timer(seconds=0.1):
    current = 0.0

    def tick():
        nonlocal current
        current += seconds
        return current

    return tick


def test_fixed_96_game_schedule_and_legal_distinct_openings():
    specs = arena.schedule()
    assert len(specs) == 96 and len({s["id"] for s in specs}) == 96
    assert specs[0]["white"] == "width128-53" and specs[0]["black"] == "width32-53"
    assert specs[-1]["white"] == "width32-83" and specs[-1]["black"] == "width128-83"
    assert Counter(s["seed"] for s in specs) == {53: 32, 67: 32, 83: 32}
    assert arena.schedule() == specs
    endpoints = set()
    for opening in arena.OPENINGS:
        board = chess.Board()
        assert len(opening["moves"]) == 6
        for uci in opening["moves"]:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
        assert len(board.move_stack) == 6 and not board.is_game_over(claim_draw=False)
        endpoints.add(board.fen())
        for seed in (53, 67, 83):
            pair = [s for s in specs if s["opening_id"] == opening["id"] and s["seed"] == seed]
            assert len(pair) == 2 and pair[0]["white"] == pair[1]["black"]
            assert pair[0]["black"] == pair[1]["white"]
    assert len(endpoints) == 16
    protocol = arena.ARENA_PROTOCOL
    assert protocol["clock_seconds"] == 300 and protocol["max_played_plies"] == 240
    assert protocol["claim_draw"] is False and protocol["depth"] == 4
    assert protocol["torch_threads"] == 2 and protocol["bootstrap_seed"] == 10800111


@pytest.mark.parametrize("fault", ["count", "duplicate", "illegal", "length"])
def test_opening_contract_rejects_mutations(fault):
    openings = copy.deepcopy(list(arena.OPENINGS))
    if fault == "count":
        openings.pop()
    elif fault == "duplicate":
        openings[1] = copy.deepcopy(openings[0])
    elif fault == "illegal":
        openings[0]["moves"][0] = "a1a8"
    else:
        openings[0]["moves"].pop()
    with pytest.raises(ValueError):
        arena.schedule(openings)


def test_prefix_history_preserved_cap_not_draw_and_board_isolated(monkeypatch):
    monkeypatch.setattr(arena, "MAX_PLAYED_PLIES", 2)
    spec = arena.schedule()[0]
    histories = []

    def policy(board):
        histories.append([m.uci() for m in board.move_stack])
        answer = response(board, spec)
        board.clear()
        return answer

    game = arena.play(spec, policy, policy, timer=timer())
    assert histories[0] == spec["opening_moves"]
    assert histories[1][:6] == spec["opening_moves"] and len(histories[1]) == 7
    assert game["status"] == "unfinished" and game["result"] == "*"
    assert game["played_plies"] == 2 and len(game["moves"]) == 8
    assert len(game["attempts"]) == 2 and game["wall_seconds"] == pytest.approx(0.2)
    assert game["clocks"] == pytest.approx({"white": 299.9, "black": 299.9})
    assert all(m["latency_ms"] == 0 for m in game["moves"][:6])
    arena.validate_game(game, spec)
    parsed = chess.pgn.read_game(io.StringIO(arena.to_pgn(game)))
    assert not parsed.errors and len(list(parsed.mainline_moves())) == 8
    assert parsed.end().board().fen() == game["final_fen"]
    source = Path(__file__).resolve().parents[1] / "scripts/render_chess_replay.py"
    module_spec = importlib.util.spec_from_file_location("capacity_replay_test", source)
    renderer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(renderer)
    assert len(renderer.normalize_game(game)["moves"]) == 8


def test_automatic_fivefold_draw_uses_full_history_without_claiming_threefold():
    spec = arena.schedule()[0]

    def repeat(board):
        move = ("f3g1", "g8f6", "g1f3", "f6g8")[(len(board.move_stack) - 6) % 4]
        return response(board, spec, move)

    game = arena.play(spec, repeat, repeat, timer=timer())
    assert game["played_plies"] == 16
    assert game["status"] == "completed" and game["result"] == "1/2-1/2"
    assert game["termination"] == "fivefold_repetition"
    arena.validate_game(game, spec)


def test_real_checkmate_is_terminal_not_a_material_adjudication():
    spec = arena.schedule()[0]
    spec["opening_moves"] = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]
    policy = lambda board: response(board, spec, "h5f7")
    game = arena.play(spec, policy, policy, timer=timer())
    assert (game["status"], game["result"], game["termination"]) == ("completed", "1-0", "checkmate")
    assert game["played_plies"] == 1
    arena.validate_game(game, spec)


def test_timeout_discards_proposal_without_fallback():
    spec = arena.schedule()[0]
    policy = lambda board: response(board, spec)
    game = arena.play(spec, policy, policy, timer=timer(301))
    assert game["status"] == "completed" and game["result"] == "0-1"
    assert game["termination"] == "timeout" and game["played_plies"] == 0
    assert len(game["attempts"]) == 1 and not game["attempts"][0]["played"]
    assert game["clocks"]["white"] == 0 and len(game["moves"]) == 6
    arena.validate_game(game, spec)


@pytest.mark.parametrize("fault", ["exception", "choice", "menu", "identity", "argmax"])
def test_policy_failures_are_unscored_and_never_retried(fault):
    spec = arena.schedule()[0]
    calls = []

    def bad(board):
        calls.append(1)
        if fault == "exception":
            raise RuntimeError("synthetic failure")
        value = response(board, spec)
        if fault == "choice":
            value["choice"] = "a1a8"
        elif fault == "menu":
            value["probabilities"].pop(next(iter(value["probabilities"])))
        elif fault == "identity":
            value["seed"] = 17
        else:
            value["choice"] = max(value["probabilities"])
        return value

    game = arena.play(spec, bad, bad, timer=timer())
    assert game["status"] == "failed" and game["result"] == "*"
    assert game["played_plies"] == 0 and len(calls) == 1
    arena.validate_game(game, spec)


def synthetic_outcomes(wins=58, draws=0, failures=0):
    games = []
    for index, spec in enumerate(arena.schedule()):
        game = {**spec, "game_id": spec["id"], "played_plies": 0, "wall_seconds": 0, "attempts": []}
        if index < wins:
            game.update(
                status="completed",
                result="1-0" if spec["white_width"] == 128 else "0-1",
                termination="checkmate",
            )
        elif index < wins + draws:
            game.update(status="completed", result="1/2-1/2", termination="fivefold_repetition")
        else:
            game.update(
                status="failed" if index < wins + draws + failures else "unfinished",
                result="*",
                termination="max_plies",
            )
        games.append(game)
    return games


def test_unresolved_point_bounds_and_opening_bootstrap_are_deterministic():
    games = synthetic_outcomes()
    summary = arena.summarize(games)
    assert summary["width128"]["score_lower_bound"] == 58 / 96
    assert summary["width128"]["score_upper_bound"] == 1
    assert summary["status_counts"] == {"completed": 58, "unfinished": 38, "failed": 0}
    assert summary["width128"]["draws"] == 0 and summary["gate_passed"]
    assert summary == arena.summarize(games)
    bootstrap = summary["opening_cluster_bootstrap"]
    assert bootstrap["openings"] == 16 and bootstrap["replicates"] == 2000
    assert bootstrap["upper_bound_interval"] == [1, 1]
    assert bootstrap["lower_bound_interval"][0] < 58 / 96 < bootstrap["lower_bound_interval"][1]
    assert summary["elo_estimate"] is None
    assert not arena.summarize(synthetic_outcomes(wins=57))["gate_passed"]
    assert not arena.summarize(synthetic_outcomes(wins=58, failures=1))["gate_passed"]
    assert arena.summarize(synthetic_outcomes(wins=56, draws=4))["width128"]["completed_points"] == 58


@pytest.mark.parametrize("fault", ["missing", "ordering", "width", "status", "result"])
def test_summary_rejects_incomplete_or_misidentified_results(fault):
    games = synthetic_outcomes()
    if fault == "missing":
        games.pop()
    elif fault == "ordering":
        games[0], games[1] = games[1], games[0]
    elif fault == "width":
        games[0]["white_width"] = 32
    elif fault == "status":
        games[0]["status"] = "unrecognized"
    else:
        games[-1]["result"] = "1/2-1/2"
    with pytest.raises(ValueError):
        arena.summarize(games)


class SyntheticModel(torch.nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.tensor(0.0))
        self.width, self.seed, self.depth, self.recurrence = width, seed, 4, "residual"

    def choose(self, board, depth=None):
        assert depth == 4 and len(board.move_stack) >= 6
        assert torch.get_num_threads() == 2
        return response(board, {"white_width": self.width, "black_width": self.width, "seed": self.seed})


@pytest.fixture
def saved_arena(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "AnchorChess", SyntheticModel)
    monkeypatch.setattr(arena, "MAX_PLAYED_PLIES", 2)
    models = {(width, seed): SyntheticModel(width, seed) for width in arena.WIDTHS for seed in arena.SEEDS}
    saved = tmp_path / "arena"
    previous = torch.get_num_threads()
    arena.run(saved, models, PLAN_HASH)
    yield saved
    torch.set_num_threads(previous)


def test_complete_saved_arena_roundtrip_and_no_overwrite(saved_arena):
    summary = arena.report(saved_arena, PLAN_HASH)
    assert summary["games"] == 96 and summary["status_counts"]["unfinished"] == 96
    assert summary["policy_calls"] == summary["played_plies"] == 192
    assert summary["width128"]["score_lower_bound"] == 0
    assert summary["width128"]["score_upper_bound"] == 1
    assert not summary["gate_passed"]
    assert arena.audit(saved_arena, PLAN_HASH) == summary
    models = {(w, s): SyntheticModel(w, s) for w in arena.WIDTHS for s in arena.SEEDS}
    with pytest.raises(FileExistsError):
        arena.run(saved_arena, models, PLAN_HASH)
    with pytest.raises(ValueError):
        arena.report(saved_arena, "b" * 64)


def rehash(saved):
    receipt = arena._read(saved / "completed.json")
    receipt["files"] = {
        name: arena._hash(saved / name) for name in receipt["files"] if (saved / name).exists()
    }
    (saved / "completed.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize(
    "fault",
    [
        "player",
        "opening",
        "clock",
        "probabilities",
        "early_stop",
        "fake_draw",
        "pgn",
        "missing",
        "summary",
        "protocol",
    ],
)
def test_rehashed_saved_corruption_is_rejected(saved_arena, fault):
    path = saved_arena / "game-001.json"
    game = arena._read(path)
    if fault == "player":
        game["white"] = "width32-53"
    elif fault == "opening":
        game["moves"][0]["latency_ms"] = 1
    elif fault == "clock":
        game["attempts"][0]["clocks"]["white"] += 1
    elif fault == "probabilities":
        game["attempts"][0]["policy"]["probabilities"].pop(
            next(iter(game["attempts"][0]["policy"]["probabilities"]))
        )
    elif fault == "early_stop":
        game["attempts"].pop()
        game["moves"].pop()
        game["played_plies"] -= 1
        game["final_fen"] = game["moves"][-1]["fen_after"]
        game["clocks"] = game["moves"][-1]["clocks"]
        game["wall_seconds"] = sum(a["wall_ms"] for a in game["attempts"]) / 1000
    elif fault == "fake_draw":
        game.update(status="completed", result="1/2-1/2", termination="threefold_repetition")
    elif fault == "pgn":
        (saved_arena / "game-001.pgn").write_text('[Result "1-0"]\n\n1-0\n')
    elif fault == "missing":
        (saved_arena / "game-096.json").unlink()
    elif fault == "summary":
        summary = arena._read(saved_arena / "summary.json")
        summary["gate_passed"] = True
        (saved_arena / "summary.json").write_text(json.dumps(summary))
    else:
        started = arena._read(saved_arena / "started.json")
        started["protocol"]["openings"][0]["name"] = "Changed after outcomes"
        (saved_arena / "started.json").write_text(json.dumps(started))
    if fault in {"player", "opening", "clock", "probabilities", "early_stop", "fake_draw"}:
        path.write_text(json.dumps(game))
    rehash(saved_arena)
    with pytest.raises(ValueError):
        arena.report(saved_arena, PLAN_HASH)


def test_wrong_model_binding_fails_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "AnchorChess", SyntheticModel)
    models = {(w, s): SyntheticModel(w, s) for w in arena.WIDTHS for s in arena.SEEDS}
    models[32, 53].recurrence = "anchor"
    with pytest.raises(ValueError):
        arena.run(tmp_path / "bad", models, PLAN_HASH)
    assert not (tmp_path / "bad").exists()
