"""Posthoc, exploratory diagnostics of the completed capacity arena.

Reads saved traces only after the frozen arena auditor accepts all 96 games.
There is no engine call, model inference, training, outcome adjudication or
change to the frozen continuation gate. Immediate mates and stalemates use
exact legal python-chess successors. Material balance is descriptive and is
not a winning probability or evidence that an alternative move was better.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path

import chess

from openjev.research import chess_capacity_arena as arena

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
SCOPE = {
    "analysis_kind": "posthoc_exploratory",
    "input": "Previously recorded, fully audited 96-game capacity arena only",
    "operations": "Replay accepted model turns, enumerate exact legal immediate successors, and count material in terminal draws",
    "opening_plies_excluded": 6,
    "engine_calls": 0,
    "model_calls": 0,
    "training_updates": 0,
    "changes_frozen_gate": False,
    "changes_game_results": False,
    "interpretation": "Descriptive failure diagnosis, not a new benchmark, continuation criterion, winning probability or proof that a non-stalemating alternative was better",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def analyze_turn(board, choice):
    """Describe one observed legal choice without evaluating position strength."""
    if not board.is_valid() or board.is_game_over(claim_draw=False):
        raise ValueError("Expected a valid nonterminal board")
    legal = sorted(move.uci() for move in board.legal_moves)
    if choice not in legal:
        raise ValueError("Recorded choice is not legal")
    mating, stalemating = [], []
    for uci in legal:
        successor = board.copy(stack=True)
        successor.push_uci(uci)
        if successor.is_checkmate():
            mating.append(uci)
        if successor.is_stalemate():
            stalemating.append(uci)
    selected_stalemate = choice in stalemating
    alternatives = [uci for uci in legal if uci not in stalemating] if selected_stalemate else []
    return {
        "fen": board.fen(),
        "legal_uci": legal,
        "choice_uci": choice,
        "mating_uci": mating,
        "mate_opportunity": bool(mating),
        "mate_hit": bool(mating) and choice in mating,
        "mate_miss": bool(mating) and choice not in mating,
        "selected_stalemate": selected_stalemate,
        "nonstalemate_alternatives_uci": alternatives,
        "selected_stalemate_with_nonstalemate_alternative": selected_stalemate and bool(alternatives),
    }


def draw_material(game):
    """Return each model's signed own-minus-opponent material at a terminal draw."""
    if game["status"] != "completed" or game["result"] != "1/2-1/2":
        return []
    board = chess.Board(game["final_fen"])
    totals = {
        color: sum(value * len(board.pieces(piece, color)) for piece, value in PIECE_VALUES.items())
        for color in (chess.WHITE, chess.BLACK)
    }
    return [
        {
            "game_id": game["game_id"],
            "opening_id": game["opening_id"],
            "seed": game["seed"],
            "model": game[side],
            "width": game[f"{side}_width"],
            "side": side,
            "final_fen": game["final_fen"],
            "termination": game["termination"],
            "own_material": totals[color],
            "opponent_material": totals[not color],
            "material_balance": totals[color] - totals[not color],
        }
        for side, color in (("white", chess.WHITE), ("black", chess.BLACK))
    ]


def analyze_game(game, spec):
    """Revalidate one complete trace, then exclude its six prescribed moves."""
    arena.validate_game(game, spec)
    board = chess.Board()
    turns = []
    for index, accepted in enumerate(game["moves"]):
        if accepted["fen_before"] != board.fen():
            raise ValueError("Broken recorded position chain")
        if index >= 6:
            side = "white" if board.turn else "black"
            turns.append(
                {
                    "game_id": game["game_id"],
                    "opening_id": game["opening_id"],
                    "seed": game["seed"],
                    "model": game[side],
                    "width": game[f"{side}_width"],
                    "side": side,
                    "ply": index + 1,
                    "model_ply": index - 5,
                    **analyze_turn(board, accepted["move_uci"]),
                }
            )
        board.push_uci(accepted["move_uci"])
        if accepted["fen_after"] != board.fen():
            raise ValueError("Incorrect legal successor")
    return turns, draw_material(game)


def summarize(turns, draws, source_summary):
    counters = (
        "mate_opportunity",
        "mate_hit",
        "mate_miss",
        "selected_stalemate",
        "selected_stalemate_with_nonstalemate_alternative",
    )
    models = {}
    for width in arena.WIDTHS:
        for seed in arena.SEEDS:
            name = arena.model_name(width, seed)
            model_turns = [row for row in turns if row["model"] == name]
            model_draws = [row for row in draws if row["model"] == name]
            values = [row["material_balance"] for row in model_draws]
            models[name] = {
                "width": width,
                "seed": seed,
                "accepted_turns": len(model_turns),
                **{key: sum(row[key] for row in model_turns) for key in counters},
                "terminal_draw_material": {
                    "games": len(values),
                    "mean_own_minus_opponent": math.fsum(values) / len(values) if values else None,
                    "positive": sum(v > 0 for v in values),
                    "zero": sum(v == 0 for v in values),
                    "negative": sum(v < 0 for v in values),
                },
            }
    return {
        "status": "completed",
        "scope": SCOPE,
        "games": source_summary["games"],
        "original_status_counts": source_summary["status_counts"],
        "accepted_model_turns": len(turns),
        "terminal_draw_games": len(draws) // 2,
        "models": models,
        "totals": {key: sum(row[key] for row in turns) for key in counters},
        "mate_miss_locations": [
            {key: row[key] for key in ("game_id", "ply", "model", "fen", "choice_uci", "mating_uci")}
            for row in turns
            if row["mate_miss"]
        ],
        "selected_stalemate_locations": [
            {
                key: row[key]
                for key in ("game_id", "ply", "model", "fen", "choice_uci", "nonstalemate_alternatives_uci")
            }
            for row in turns
            if row["selected_stalemate_with_nonstalemate_alternative"]
        ],
        "draw_termination_counts": dict(
            Counter(row["termination"] for row in draws if row["side"] == "white")
        ),
        "material_values": {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9},
    }


def analyze(saved, out, plan_hash):
    """Create new diagnostic artifacts only after the frozen full-panel audit."""
    saved, out = Path(saved), Path(out)
    source_summary = arena.report(saved, plan_hash)
    specs = arena.schedule()
    if len(specs) != 96 or source_summary["games"] != 96:
        raise ValueError("Diagnostics require the complete fixed 96-game arena")
    receipt = read(saved / "completed.json")
    files = {name: sha(saved / name) for name in sorted([*receipt["files"], "completed.json"])}
    code = {
        str(Path(__file__).resolve()): sha(__file__),
        str(Path(arena.__file__).resolve()): sha(arena.__file__),
    }
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Scope is written before successor enumeration; this is not a new efficacy test.
    write_new(
        out / "scope.json",
        {
            **SCOPE,
            "plan_sha256": plan_hash,
            "code_sha256": code,
            "source_arena": str(saved.resolve()),
            "source_files": files,
        },
    )
    turns, draws = [], []
    with (out / "turns.jsonl").open("x") as turn_stream, (out / "draws.jsonl").open("x") as draw_stream:
        for spec in specs:
            game_turns, game_draws = analyze_game(read(saved / f"{spec['id']}.json"), spec)
            turns.extend(game_turns)
            draws.extend(game_draws)
            for stream, values in ((turn_stream, game_turns), (draw_stream, game_draws)):
                for row in values:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    if any(sha(saved / name) != digest for name, digest in files.items()):
        raise ValueError("Source arena changed during the diagnostic")
    summary = summarize(turns, draws, source_summary)
    if len(turns) != source_summary["played_plies"]:
        raise ValueError("Accepted model-turn membership differs from the audited arena")
    write_new(out / "summary.json", summary)
    write_new(
        out / "completed.json",
        {
            "status": "completed",
            "analysis_kind": "posthoc_exploratory",
            "plan_sha256": plan_hash,
            "source_arena": str(saved.resolve()),
            "source_files": files,
            "source_game_sha256": {f"{s['id']}.json": files[f"{s['id']}.json"] for s in specs},
            "code_sha256": code,
            "dependencies": {"chess": importlib.metadata.version("chess")},
            "wall_seconds": time.perf_counter() - started,
            "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = analyze(args.arena, args.out, sha(args.plan))
    print(
        json.dumps(
            {"status": summary["status"], "games": summary["games"], "totals": summary["totals"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
