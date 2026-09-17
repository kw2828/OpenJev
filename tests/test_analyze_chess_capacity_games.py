"""Small native-board fixtures for the separate posthoc arena diagnostic."""

import copy
import importlib.util
import json
from pathlib import Path

import chess
import pytest

from openjev.research import chess_capacity_arena as arena

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capacity_games_diagnostic", ROOT / "scripts/analyze_chess_capacity_games.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
PLAN_HASH = "a" * 64


def policy(spec, move=None):
    def choose(board):
        candidates = sorted(m.uci() for m in board.legal_moves)
        chosen = move or candidates[0]
        side = "white" if board.turn else "black"
        return {
            "choice": chosen,
            "probabilities": {uci: float(uci == chosen) for uci in candidates},
            "width": spec[f"{side}_width"],
            "seed": spec["seed"],
            "depth": 4,
            "mode": "recurrent",
            "recurrence": "residual",
        }

    return choose


def test_exact_mate_opportunity_hit_and_stalemate_miss():
    board = chess.Board("k7/8/1QK5/8/8/8/8/8 w - - 0 1")
    mate = diagnostic.analyze_turn(board, "b6b7")
    assert mate["mate_opportunity"] and mate["mate_hit"] and not mate["mate_miss"]
    assert "b6b7" in mate["mating_uci"] and not mate["selected_stalemate"]
    missed = diagnostic.analyze_turn(board, "b6c7")
    assert missed["mate_opportunity"] and missed["mate_miss"] and not missed["mate_hit"]
    assert missed["selected_stalemate_with_nonstalemate_alternative"]
    assert "b6b7" in missed["nonstalemate_alternatives_uci"]
    assert "b6c7" not in missed["nonstalemate_alternatives_uci"]
    assert set(missed["legal_uci"]) == {m.uci() for m in board.legal_moves}
    assert missed["fen"] == board.fen()


def mate_game():
    spec = arena.schedule()[0]
    spec["opening_moves"] = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]
    choose = policy(spec, "h5f7")
    game = arena.play(spec, choose, choose, timer=lambda: 0.0)
    return game, spec


def test_prescribed_six_plies_are_skipped_and_recorded_fen_is_exact():
    game, spec = mate_game()
    turns, draws = diagnostic.analyze_game(game, spec)
    assert len(game["moves"]) == 7 and len(turns) == 1 and draws == []
    turn = turns[0]
    assert turn["ply"] == 7 and turn["model_ply"] == 1
    assert turn["game_id"] == spec["id"] and turn["model"] == "width128-53"
    assert turn["fen"] == game["moves"][6]["fen_before"]
    assert turn["choice_uci"] == "h5f7" and turn["mate_hit"]


@pytest.mark.parametrize("fault", ["fen", "uci", "result"])
def test_game_diagnostics_refuse_invalid_replays(fault):
    game, spec = mate_game()
    if fault == "fen":
        game["moves"][0]["fen_after"] = chess.STARTING_FEN
    elif fault == "uci":
        game["moves"][-1]["move_uci"] = "a1a8"
    else:
        game["result"] = "1/2-1/2"
    with pytest.raises(ValueError):
        diagnostic.analyze_game(game, spec)


def test_terminal_draw_material_is_oriented_by_player_not_width():
    spec = arena.schedule()[0]
    game = {
        **spec,
        "game_id": spec["id"],
        "status": "completed",
        "result": "1/2-1/2",
        "termination": "stalemate",
        "final_fen": "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
    }
    rows = diagnostic.draw_material(game)
    assert [(row["width"], row["material_balance"]) for row in rows] == [(128, 9), (32, -9)]
    game.update(white="width32-53", black="width128-53", white_width=32, black_width=128)
    rows = diagnostic.draw_material(game)
    assert [(row["width"], row["material_balance"]) for row in rows] == [(32, 9), (128, -9)]
    game.update(status="unfinished", result="*")
    assert diagnostic.draw_material(game) == []


def test_invalid_choices_and_terminal_boards_are_not_diagnosed():
    with pytest.raises(ValueError):
        diagnostic.analyze_turn(chess.Board(), "a1a8")
    with pytest.raises(ValueError):
        diagnostic.analyze_turn(chess.Board("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1"), "a8a7")


@pytest.fixture
def synthetic_arena(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "MAX_PLAYED_PLIES", 2)
    saved = tmp_path / "saved"
    saved.mkdir()
    models = {
        arena.model_name(w, s): {
            "width": w,
            "seed": s,
            "depth": 4,
            "recurrence": "residual",
            "state_sha256": "b" * 64,
        }
        for w in arena.WIDTHS
        for s in arena.SEEDS
    }
    diagnostic.write_new(
        saved / "started.json",
        {"status": "started", "plan_sha256": PLAN_HASH, "protocol": arena.protocol(), "models": models},
    )
    games = []
    for spec in arena.schedule():
        choose = policy(spec)
        game = arena.play(spec, choose, choose, timer=lambda: 0.0)
        diagnostic.write_new(saved / f"{spec['id']}.json", game)
        (saved / f"{spec['id']}.pgn").write_text(arena.to_pgn(game))
        games.append(game)
    diagnostic.write_new(saved / "summary.json", arena.summarize(games))
    diagnostic.write_new(
        saved / "completed.json",
        {
            "status": "completed",
            "plan_sha256": PLAN_HASH,
            "wall_seconds": 0.0,
            "files": {p.name: diagnostic.sha(p) for p in sorted(saved.iterdir())},
        },
    )
    return saved


def test_complete_diagnostic_preserves_source_and_binds_all_turns(synthetic_arena, tmp_path):
    saved = synthetic_arena
    before = {p.name: diagnostic.sha(p) for p in saved.iterdir()}
    out = tmp_path / "diagnostic"
    summary = diagnostic.analyze(saved, out, PLAN_HASH)
    assert summary["games"] == 96 and summary["accepted_model_turns"] == 192
    turns = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == 192 and {row["ply"] for row in turns} == {7, 8}
    assert summary["scope"]["analysis_kind"] == "posthoc_exploratory"
    assert summary["scope"]["engine_calls"] == summary["scope"]["model_calls"] == 0
    assert not summary["scope"]["changes_game_results"] and not summary["scope"]["changes_frozen_gate"]
    receipt = diagnostic.read(out / "completed.json")
    assert len(receipt["source_game_sha256"]) == 96
    assert receipt["source_files"] == before
    assert all(diagnostic.sha(out / name) == digest for name, digest in receipt["files"].items())
    assert {p.name: diagnostic.sha(p) for p in saved.iterdir()} == before
    assert str(Path(diagnostic.__file__).resolve()) in receipt["code_sha256"]
    with pytest.raises(FileExistsError):
        diagnostic.analyze(saved, out, PLAN_HASH)


def test_source_audit_rejection_occurs_before_output_creation(synthetic_arena, tmp_path):
    saved = synthetic_arena
    path = saved / "game-001.json"
    changed = diagnostic.read(path)
    changed["result"] = "1-0"
    path.write_text(json.dumps(changed))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        diagnostic.analyze(saved, out, PLAN_HASH)
    assert not out.exists()


def test_summary_uses_all_six_models_without_claiming_winning_probability():
    game, spec = mate_game()
    turns, _ = diagnostic.analyze_game(game, spec)
    draw = copy.deepcopy(game)
    draw.update(
        status="completed",
        result="1/2-1/2",
        termination="stalemate",
        final_fen="k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
    )
    result = diagnostic.summarize(
        turns,
        diagnostic.draw_material(draw),
        {"games": 96, "status_counts": {"completed": 1, "unfinished": 95, "failed": 0}},
    )
    assert len(result["models"]) == 6
    assert result["models"]["width128-53"]["mate_hit"] == 1
    assert result["models"]["width128-53"]["terminal_draw_material"]["mean_own_minus_opponent"] == 9
    assert result["models"]["width32-53"]["terminal_draw_material"]["mean_own_minus_opponent"] == -9
    assert result["models"]["width128-67"]["terminal_draw_material"]["mean_own_minus_opponent"] is None
    assert "gate_passed" not in result and "winning_probability" not in result
