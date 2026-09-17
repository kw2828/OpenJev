"""Paired capacity arena with complete opening history and unresolved-game bounds.

The sixteen common opening prefixes are selected before outcomes and paired by
color. They have not been engine-certified as equal positions. Every move after
the six-ply prefix is an unchanged depth-four model decision, without search,
fallback or a tactical guard. Capped games remain unfinished, never draws.
"""

import hashlib
import io
import json
import math
import time
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_arena import GameResult, _response

SEEDS = (53, 67, 83)
WIDTHS = (32, 128)
CLOCK_SECONDS = 300.0
MAX_PLAYED_PLIES = 240
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 10800111
OPENINGS = tuple(
    {"id": f"opening-{i:02d}", "name": name, "moves": moves.split()}
    for i, (name, moves) in enumerate(
        (
            ("Ruy Lopez", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"),
            ("Italian Game", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
            ("Scotch Game", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4"),
            ("Petrov Defense", "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6"),
            ("Philidor Defense", "e2e4 e7e5 g1f3 d7d6 d2d4 g8f6"),
            ("Sicilian Defense", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4"),
            ("French Defense", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6"),
            ("Caro-Kann Defense", "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4"),
            ("Queen's Gambit Declined", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6"),
            ("Slav Defense", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6"),
            ("Queen's Gambit Accepted", "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6"),
            ("King's Indian Defense", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7"),
            ("Nimzo-Indian Defense", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4"),
            ("Grunfeld Defense", "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5"),
            ("English Opening", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6"),
            ("Reti Opening", "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6"),
        ),
        1,
    )
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid_constant)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _plan_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA256 plan hash")


def model_name(width, seed):
    return f"width{width}-{seed}"


def _openings(openings):
    values = json.loads(json.dumps(openings, allow_nan=False))
    if len(values) != 16 or len({o["id"] for o in values}) != 16:
        raise ValueError("Exactly sixteen distinct opening IDs are required")
    endpoints = set()
    for opening in values:
        if set(opening) != {"id", "name", "moves"} or not opening["name"] or len(opening["moves"]) != 6:
            raise ValueError("Every opening requires a name and exactly six UCI moves")
        board = chess.Board()
        for uci in opening["moves"]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves or board.is_game_over(claim_draw=False):
                raise ValueError("Opening prefix contains an illegal or post-terminal move")
            board.push(move)
        if board.is_game_over(claim_draw=False) or board.fen() in endpoints:
            raise ValueError("Opening endpoints must be distinct and nonterminal")
        endpoints.add(board.fen())
    return values


def schedule(openings=OPENINGS):
    """Return all 16 openings x 3 paired seeds x 2 color assignments in order."""
    games = []
    for opening in _openings(openings):
        for seed in SEEDS:
            for white_width, black_width in ((128, 32), (32, 128)):
                games.append(
                    {
                        "id": f"game-{len(games) + 1:03d}",
                        "opening_id": opening["id"],
                        "opening_name": opening["name"],
                        "opening_moves": list(opening["moves"]),
                        "seed": seed,
                        "white": model_name(white_width, seed),
                        "black": model_name(black_width, seed),
                        "white_width": white_width,
                        "black_width": black_width,
                    }
                )
    return games


def protocol(openings=OPENINGS):
    return {
        "openings": _openings(openings),
        "games": schedule(openings),
        "clock_seconds": CLOCK_SECONDS,
        "increment": 0,
        "max_played_plies": MAX_PLAYED_PLIES,
        "claim_draw": False,
        "device": "cpu",
        "torch_threads": 2,
        "depth": 4,
        "initial_fen": chess.STARTING_FEN,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate_lower_score": 0.60,
        "scope": "Paired common-opening development games, conditional on three fitted seeds; no Elo or broad strength claim",
        "opening_scope": "Six prescribed legal plies, no clock debit; common openings, not engine-certified balance",
        "clock_scope": "Synchronous CPU policy call wall time; no increment, retries, warmups or fallback; model loading excluded",
        "timeout_scope": "Clock expiry loses unless python-chess recognizes the opponent's material as insufficient; calls are timed on return",
    }


def _opening_rows(spec):
    board = chess.Board()
    clocks = {"white": CLOCK_SECONDS, "black": CLOCK_SECONDS}
    result = []
    for index, uci in enumerate(spec["opening_moves"], 1):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("Illegal opening move")
        side = "white" if board.turn else "black"
        before, san = board.fen(), board.san(move)
        board.push(move)
        result.append(
            {
                "ply": index,
                "side": side,
                "fen_before": before,
                "fen_after": board.fen(),
                "move_uci": uci,
                "san": san,
                "latency_ms": 0.0,
                "wall_ms": 0.0,
                "clocks_before": dict(clocks),
                "clocks": dict(clocks),
                "policy": None,
                "played": True,
                "disposition": "prescribed_opening",
            }
        )
    return board, result


def _parse_response(response, board):
    value = _response(response, {m.uci() for m in board.legal_moves})
    if "probabilities" not in value:
        raise ValueError("Capacity models must return all legal candidate probabilities")
    if value["choice"] in value["probabilities"] and value["choice"] != max(
        sorted(value["probabilities"]), key=value["probabilities"].get
    ):
        raise ValueError("Capacity model choice must be the unchanged probability argmax")
    return value


def _identity(response, spec, side):
    expected = {
        "width": spec[f"{side}_width"],
        "seed": spec["seed"],
        "depth": 4,
        "mode": "recurrent",
        "recurrence": "residual",
    }
    if any(response.get(key) != value for key, value in expected.items()):
        raise ValueError("Response model identity does not match the scheduled player")


def play(spec, white, black, *, timer=None):
    """Run one scheduled game. A test timer may replace the monotonic clock."""
    board, opening = _opening_rows(spec)
    clock = time.perf_counter if timer is None else timer
    initial = {"white": CLOCK_SECONDS, "black": CLOCK_SECONDS}
    result = GameResult(
        chess.STARTING_FEN,
        spec["white"],
        spec["black"],
        initial,
        dict(initial),
        False,
        MAX_PLAYED_PLIES,
        moves=opening,
    )
    while True:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            result.status, result.result = "completed", outcome.result()
            result.termination = outcome.termination.name.lower()
            break
        if len(result.moves) - 6 >= MAX_PLAYED_PLIES:
            break
        side = "white" if board.turn else "black"
        before, clocks_before = board.fen(), dict(result.clocks)
        response, error, validation = None, None, None
        start = clock()
        try:
            response = (white if board.turn else black)(board.copy(stack=True))
        except Exception as exc:  # noqa: BLE001 - retain policy failure without substitution
            error = {"type": type(exc).__name__, "message": str(exc)}
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Timer must produce finite monotonic elapsed times")
        result.clocks[side] = max(0.0, result.clocks[side] - elapsed)
        parsed = None
        if error is None:
            try:
                parsed = _parse_response(response, board)
                _identity(parsed, spec, side)
            except (TypeError, ValueError, OverflowError) as exc:
                validation = {"type": type(exc).__name__, "message": str(exc)}
        # Preserve a JSON-safe invalid response when possible, including illegal choices.
        snapshot = None
        try:
            snapshot = json.loads(json.dumps(response, allow_nan=False))
        except (TypeError, ValueError, OverflowError):
            pass
        attempt = {
            "ply": len(result.moves) + 1,
            "side": side,
            "fen_before": before,
            "fen_after": before,
            "move_uci": snapshot.get("choice") if isinstance(snapshot, dict) else None,
            "san": None,
            "latency_ms": elapsed * 1000,
            "wall_ms": elapsed * 1000,
            "clocks_before": clocks_before,
            "clocks": dict(result.clocks),
            "policy": snapshot,
            "played": False,
        }
        if error is not None or validation is not None:
            attempt["error"] = error or validation
        result.attempts.append(attempt)
        if result.clocks[side] <= 0:
            draw = board.has_insufficient_material(not board.turn)
            result.status = "completed"
            result.result = "1/2-1/2" if draw else "0-1" if board.turn else "1-0"
            result.termination = "timeout_insufficient_material" if draw else "timeout"
            attempt["disposition"] = "discarded_timeout"
            break
        if error is not None or validation is not None:
            result.status = "failed"
            result.termination = "policy_error" if error is not None else "invalid_response"
            attempt["disposition"] = result.termination
            break
        if parsed["choice"] not in {m.uci() for m in board.legal_moves}:
            result.status, result.termination = "failed", "invalid_choice"
            attempt["disposition"] = "invalid_choice"
            break
        move = chess.Move.from_uci(parsed["choice"])
        attempt["san"] = board.san(move)
        board.push(move)
        attempt.update(fen_after=board.fen(), played=True, disposition="played")
        result.moves.append(dict(attempt))
    result.final_fen = board.fen()
    game = {
        **result.to_dict(),
        "game_id": spec["id"],
        "opening_id": spec["opening_id"],
        "opening_name": spec["opening_name"],
        "opening_moves": spec["opening_moves"],
        "seed": spec["seed"],
        "white_width": spec["white_width"],
        "black_width": spec["black_width"],
        "opening_plies": 6,
        "played_plies": len(result.moves) - 6,
        "wall_seconds": math.fsum(a["wall_ms"] for a in result.attempts) / 1000,
        "max_plies_scope": "Model-played plies after the six prescribed opening plies",
    }
    return game


def to_pgn(game):
    value = GameResult(**{name: game[name] for name in GameResult.__dataclass_fields__})
    return value.to_pgn() + "\n"


def validate_game(game, spec):
    """Independently rebuild full history, each policy attempt and final outcome."""
    for key in (
        "white",
        "black",
        "opening_id",
        "opening_name",
        "opening_moves",
        "seed",
        "white_width",
        "black_width",
    ):
        if game[key] != spec[key]:
            raise ValueError("Scheduled player or opening identity mismatch")
    initial = {"white": CLOCK_SECONDS, "black": CLOCK_SECONDS}
    if (
        game["game_id"] != spec["id"]
        or game["initial_fen"] != chess.STARTING_FEN
        or game["claim_draw"] is not False
        or game["max_plies"] != MAX_PLAYED_PLIES
        or game["opening_plies"] != 6
        or game["initial_clocks"] != initial
        or game["max_plies_scope"] != "Model-played plies after the six prescribed opening plies"
    ):
        raise ValueError("Game rules or identity mismatch")
    board, played = _opening_rows(spec)
    if game["moves"][:6] != played:
        raise ValueError("Prescribed opening trace changed")
    clocks = dict(initial)
    for index, attempt in enumerate(game["attempts"]):
        side = "white" if board.turn else "black"
        if (
            board.outcome(claim_draw=False) is not None
            or len(played) - 6 >= MAX_PLAYED_PLIES
            or attempt["fen_before"] != board.fen()
            or attempt["side"] != side
            or attempt["ply"] != len(played) + 1
            or attempt["clocks_before"] != clocks
        ):
            raise ValueError("Attempt violates position history, clock linkage or stop rule")
        elapsed = attempt["latency_ms"] / 1000
        if not math.isfinite(elapsed) or elapsed < 0 or attempt["wall_ms"] != attempt["latency_ms"]:
            raise ValueError("Invalid elapsed policy time")
        clocks[side] = max(0.0, clocks[side] - elapsed)
        if set(attempt["clocks"]) != set(clocks) or any(
            not math.isfinite(attempt["clocks"][s])
            or not math.isclose(attempt["clocks"][s], clocks[s], abs_tol=1e-9, rel_tol=0)
            for s in clocks
        ):
            raise ValueError("Incorrect clock debit")
        clocks = dict(attempt["clocks"])
        snapshot = attempt["policy"]
        if attempt["move_uci"] != (snapshot.get("choice") if isinstance(snapshot, dict) else None):
            raise ValueError("Attempt choice differs from the saved policy response")
        if attempt["played"] is True:
            parsed = _parse_response(attempt["policy"], board)
            _identity(parsed, spec, side)
            move = chess.Move.from_uci(attempt["move_uci"])
            if (
                move not in board.legal_moves
                or parsed["choice"] != attempt["move_uci"]
                or attempt["san"] != board.san(move)
                or attempt["disposition"] != "played"
                or clocks[side] <= 0
                or "error" in attempt
            ):
                raise ValueError("Invalid accepted policy move")
            board.push(move)
            played.append(attempt)
        elif (
            attempt["played"] is not False or index != len(game["attempts"]) - 1 or attempt["san"] is not None
        ):
            raise ValueError("Invalid discarded attempt or play after failure")
        if attempt["fen_after"] != board.fen():
            raise ValueError("Wrong exact successor")
    if (
        played != game["moves"]
        or game["final_fen"] != board.fen()
        or game["clocks"] != clocks
        or game["played_plies"] != len(played) - 6
        or not math.isclose(
            game["wall_seconds"],
            math.fsum(a["wall_ms"] for a in game["attempts"]) / 1000,
            abs_tol=1e-9,
            rel_tol=0,
        )
    ):
        raise ValueError("Final trace or cost mismatch")
    outcome = board.outcome(claim_draw=False)
    last = game["attempts"][-1] if game["attempts"] else None
    if outcome is not None:
        expected = ("completed", outcome.result(), outcome.termination.name.lower())
    elif last is not None and not last["played"] and clocks[last["side"]] == 0:
        draw = board.has_insufficient_material(not board.turn)
        expected = (
            "completed",
            "1/2-1/2" if draw else "0-1" if board.turn else "1-0",
            "timeout_insufficient_material" if draw else "timeout",
        )
        if last["disposition"] != "discarded_timeout":
            raise ValueError("Missing timeout disposition")
    elif last is not None and not last["played"]:
        reason = last["disposition"]
        if reason == "policy_error":
            if (
                not isinstance(last.get("error"), dict)
                or set(last["error"]) != {"type", "message"}
                or not all(isinstance(v, str) for v in last["error"].values())
                or last["policy"] is not None
            ):
                raise ValueError("Missing recorded policy exception")
        elif reason == "invalid_response":
            try:
                _parse_response(last["policy"], board)
                _identity(last["policy"], spec, last["side"])
            except (TypeError, ValueError, OverflowError):
                pass
            else:
                raise ValueError("Valid response mislabeled invalid")
        elif reason == "invalid_choice":
            parsed = _parse_response(last["policy"], board)
            if parsed["choice"] in {m.uci() for m in board.legal_moves}:
                raise ValueError("Legal choice mislabeled invalid")
        else:
            raise ValueError("Unknown policy failure")
        expected = ("failed", "*", reason)
    elif len(played) - 6 == MAX_PLAYED_PLIES:
        expected = ("unfinished", "*", "max_plies")
    else:
        raise ValueError("Game stopped without a terminal result or cap")
    if (game["status"], game["result"], game["termination"]) != expected:
        raise ValueError("Terminal status, result or reason does not match replay")
    return game


def summarize(games, openings=OPENINGS):
    """Bound width-128 points over the complete panel; never score caps as draws."""
    specs = schedule(openings)
    if len(games) != len(specs) or [g["game_id"] for g in games] != [s["id"] for s in specs]:
        raise ValueError("Summary requires the full ordered schedule")
    for game, spec in zip(games, specs, strict=True):
        if (
            any(
                game[key] != spec[key]
                for key in ("opening_id", "seed", "white", "black", "white_width", "black_width")
            )
            or game["status"] not in {"completed", "unfinished", "failed"}
            or game["result"] not in ({"1-0", "0-1", "1/2-1/2"} if game["status"] == "completed" else {"*"})
        ):
            raise ValueError("Summary player identity or terminal status mismatch")
    counts = Counter(g["status"] for g in games)
    wins = draws = losses = 0
    clusters = {o["id"]: [] for o in openings}
    for game in games:
        if game["status"] == "completed":
            if game["result"] == "1/2-1/2":
                points = 0.5
                draws += 1
            elif game["result"] == ("1-0" if game["white_width"] == 128 else "0-1"):
                points = 1.0
                wins += 1
            else:
                points = 0.0
                losses += 1
            clusters[game["opening_id"]].append((points, points))
        else:
            clusters[game["opening_id"]].append((0.0, 1.0))
    completed_points = wins + draws / 2
    unresolved = counts["unfinished"] + counts["failed"]
    lower, upper = completed_points / len(games), (completed_points + unresolved) / len(games)
    values = np.array([np.sum(cluster, axis=0) for cluster in clusters.values()])
    sample = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    boot = values[sample].sum(axis=1) / len(games)
    return {
        "status": "completed",
        "games": len(games),
        "status_counts": {s: counts[s] for s in ("completed", "unfinished", "failed")},
        "width128": {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "completed_points": completed_points,
            "possible_points": len(games),
            "score_lower_bound": lower,
            "score_upper_bound": upper,
        },
        "opening_cluster_bootstrap": {
            "openings": len(values),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "lower_bound_interval": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
            "upper_bound_interval": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
            "scope": "Resample whole openings with both colors and all three fitted seeds; conditional on those seeds. Intervals describe score bounds, not imputed unresolved outcomes.",
        },
        "gate": {
            "threshold": 0.60,
            "lower_score_bound": lower,
            "zero_failed_games": counts["failed"] == 0,
            "passed": lower >= 0.60 and counts["failed"] == 0,
        },
        "gate_passed": lower >= 0.60 and counts["failed"] == 0,
        "policy_calls": sum(len(g["attempts"]) for g in games),
        "played_plies": sum(g["played_plies"] for g in games),
        "policy_wall_seconds": math.fsum(g["wall_seconds"] for g in games),
        "game_results": [
            {
                k: g[k]
                for k in (
                    "game_id",
                    "opening_id",
                    "seed",
                    "white",
                    "black",
                    "status",
                    "result",
                    "termination",
                    "played_plies",
                )
            }
            for g in games
        ],
        "elo_estimate": None,
        "scope": protocol(openings)["scope"],
    }


def _model_bindings(models):
    if set(models) != {(width, seed) for width in WIDTHS for seed in SEEDS}:
        raise ValueError("Expected exactly the six paired width/seed models")
    result = {}
    for (width, seed), model in models.items():
        if (
            not isinstance(model, AnchorChess)
            or model.recurrence != "residual"
            or model.width != width
            or model.seed != seed
            or model.depth != 4
            or any(p.device.type != "cpu" for p in model.parameters())
        ):
            raise ValueError("Model identity, recurrence, depth or device mismatch")
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[model_name(width, seed)] = {
            "width": width,
            "seed": seed,
            "depth": 4,
            "recurrence": "residual",
            "state_sha256": digest.hexdigest(),
        }
    return result


def run(out, models, plan_hash, openings=OPENINGS):
    """Execute all scheduled games once using caller-loaded, plan-bound models."""
    _plan_hash(plan_hash)
    bindings = _model_bindings(models)
    rules = protocol(openings)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    _write(
        out / "started.json",
        {"status": "started", "plan_sha256": plan_hash, "protocol": rules, "models": bindings},
    )
    started = time.perf_counter()
    games = []
    for spec in rules["games"]:
        white = models[spec["white_width"], spec["seed"]]
        black = models[spec["black_width"], spec["seed"]]
        game = play(spec, lambda b, m=white: m.choose(b, depth=4), lambda b, m=black: m.choose(b, depth=4))
        _write(out / f"{spec['id']}.json", game)
        with (out / f"{spec['id']}.pgn").open("x") as stream:
            stream.write(to_pgn(game))
        validate_game(game, spec)
        games.append(game)
    if _model_bindings(models) != bindings:
        raise ValueError("Models changed during the arena")
    summary = summarize(games, openings)
    _write(out / "summary.json", summary)
    _write(
        out / "completed.json",
        {
            "status": "completed",
            "plan_sha256": plan_hash,
            "wall_seconds": time.perf_counter() - started,
            "files": {p.name: _hash(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    return summary


def report(saved, plan_hash, openings=OPENINGS):
    """Audit every saved game, PGN, clock and aggregate without model inference."""
    _plan_hash(plan_hash)
    saved = Path(saved)
    receipt, started = _read(saved / "completed.json"), _read(saved / "started.json")
    if (
        receipt["status"] != "completed"
        or started["status"] != "started"
        or receipt["plan_sha256"] != plan_hash
        or started["plan_sha256"] != plan_hash
        or not math.isfinite(receipt["wall_seconds"])
        or receipt["wall_seconds"] < 0
    ):
        raise ValueError("Incomplete or incorrectly bound arena")
    rules = protocol(openings)
    if started["protocol"] != rules:
        raise ValueError("Frozen arena protocol changed")
    expected_models = {model_name(w, s): (w, s) for w in WIDTHS for s in SEEDS}
    if set(started["models"]) != set(expected_models):
        raise ValueError("Incomplete model identities")
    for name, (width, seed) in expected_models.items():
        binding = started["models"][name]
        _plan_hash(binding["state_sha256"])
        if binding != {
            "width": width,
            "seed": seed,
            "depth": 4,
            "recurrence": "residual",
            "state_sha256": binding["state_sha256"],
        }:
            raise ValueError("Incorrect model binding")
    expected = {"started.json", "summary.json"} | {
        f"{spec['id']}.{ext}" for spec in rules["games"] for ext in ("json", "pgn")
    }
    if set(receipt["files"]) != expected or {p.name for p in saved.iterdir()} != expected | {
        "completed.json"
    }:
        raise ValueError("Arena files do not cover the exact fixed schedule")
    for name, digest in receipt["files"].items():
        if _hash(saved / name) != digest:
            raise ValueError("Arena evidence hash changed")
    games = []
    for spec in rules["games"]:
        game = validate_game(_read(saved / f"{spec['id']}.json"), spec)
        text = (saved / f"{spec['id']}.pgn").read_text()
        pgn = chess.pgn.read_game(io.StringIO(text))
        if (
            text != to_pgn(game)
            or pgn is None
            or pgn.errors
            or pgn.end().board().fen() != game["final_fen"]
            or [m.uci() for m in pgn.mainline_moves()] != [m["move_uci"] for m in game["moves"]]
        ):
            raise ValueError("PGN does not preserve the exact game trace")
        games.append(game)
    summary = summarize(games, rules["openings"])
    if summary != _read(saved / "summary.json"):
        raise ValueError("Arena summary does not match replayed results")
    return summary


audit = report
ARENA_PROTOCOL = protocol()
