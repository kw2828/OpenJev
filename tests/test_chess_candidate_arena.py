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

from openjev.research import chess_candidate_arena as arena

PLAN_HASH = "a" * 64


def response(board, spec, choice=None):
    legal = sorted(m.uci() for m in board.legal_moves)
    choice = legal[0] if choice is None else choice
    side = "white" if board.turn else "black"
    arm = spec[f"{side}_arm"]
    refinement = 0 if arm == "direct" else len(legal) * 2
    child_root = len(legal) * 4 if arm == "full_afterstate" else 0
    return {
        "choice": choice,
        "probabilities": {uci: float(uci == choice) for uci in legal},
        "width": 32,
        "arm": arm,
        "branch_depth": 2,
        "seed": spec["seed"],
        "depth": 4,
        "mode": "recurrent",
        "recurrence": "residual",
        "candidate_evaluations": len(legal),
        "candidate_branch_evaluations": 0 if arm == "direct" else len(legal),
        "native_successors": len(legal) if arm in ("delta", "full_afterstate") else 0,
        "root_core_iterations": 4,
        "candidate_refinement_iterations": refinement,
        "successor_root_iterations": child_root,
        "total_core_iterations": 4 + refinement + child_root,
    }


def timer(seconds=0.1):
    current = 0.0

    def tick():
        nonlocal current
        current += seconds
        return current

    return tick


def test_fixed_288_game_schedule_and_legal_distinct_openings():
    specs = arena.schedule()
    assert len(specs) == 288 and len({s["id"] for s in specs}) == 288
    assert specs[0]["white"] == "delta-97" and specs[0]["black"] == "direct-97"
    assert specs[-1]["white"] == "full_afterstate-127" and specs[-1]["black"] == "delta-127"
    assert Counter(s["seed"] for s in specs) == {97: 96, 109: 96, 127: 96}
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
        for seed in (97, 109, 127):
            pair = [
                s
                for s in specs
                if s["opening_id"] == opening["id"] and s["seed"] == seed and s["opponent"] == "direct"
            ]
            assert len(pair) == 2 and pair[0]["white"] == pair[1]["black"]
            assert pair[0]["black"] == pair[1]["white"]
    assert len(endpoints) == 16
    protocol = arena.ARENA_PROTOCOL
    assert protocol["clock_seconds"] == 300 and protocol["max_played_plies"] == 240
    assert protocol["claim_draw"] is False and protocol["depth"] == 4
    assert protocol["torch_threads"] == 2 and protocol["bootstrap_seed"] == 11300071
    assert Counter(s["opponent"] for s in specs) == dict.fromkeys(arena.OPPONENTS, 96)
    assert specs[96]["opponent"] == "action_only" and specs[192]["opponent"] == "full_afterstate"
    from openjev.research.chess_capacity_arena import OPENINGS as capacity_openings

    assert arena.OPENINGS == capacity_openings


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
    module_spec = importlib.util.spec_from_file_location("candidate_replay_test", source)
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


@pytest.mark.parametrize("fault", ["exception", "choice", "menu", "identity", "argmax", "cost"])
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
        elif fault == "cost":
            value["native_successors"] = 0
        else:
            value["choice"] = max(value["probabilities"])
        return value

    game = arena.play(spec, bad, bad, timer=timer())
    assert game["status"] == "failed" and game["result"] == "*"
    assert game["played_plies"] == 0 and len(calls) == 1
    arena.validate_game(game, spec)


def synthetic_outcomes(wins=None, draws=None, failures=None):
    wins = {o: 58 for o in arena.OPPONENTS} if wins is None else wins
    draws, failures = draws or {}, failures or {}
    games = []
    counts = Counter()
    for spec in arena.schedule():
        opponent = spec["opponent"]
        index = counts[opponent]
        counts[opponent] += 1
        w, d, f = wins.get(opponent, 0), draws.get(opponent, 0), failures.get(opponent, 0)
        game = {**spec, "game_id": spec["id"], "played_plies": 0, "wall_seconds": 0, "attempts": []}
        if index < w:
            game.update(
                status="completed",
                result="1-0" if spec["white_arm"] == "delta" else "0-1",
                termination="checkmate",
            )
        elif index < w + d:
            game.update(status="completed", result="1/2-1/2", termination="fivefold_repetition")
        else:
            game.update(
                status="failed" if index < w + d + f else "unfinished", result="*", termination="max_plies"
            )
        games.append(game)
    return games


def test_independent_primary_gates_unresolved_bounds_and_deterministic_bootstrap():
    games = synthetic_outcomes(wins={"direct": 58, "action_only": 58, "full_afterstate": 0})
    summary = arena.summarize(games)
    assert summary["gate_passed"] and summary["gate"]["zero_failed_games"]
    assert summary["status_counts"] == {"completed": 116, "unfinished": 172, "failed": 0}
    assert summary["aggregate"]["delta"]["score_lower_bound"] == 116 / 288
    assert summary["aggregate"]["delta"]["score_upper_bound"] == 1
    assert summary == arena.summarize(games)
    for opponent in arena.OPPONENTS:
        panel = summary["by_opponent"][opponent]
        assert panel["games"] == 96 and panel["delta"]["draws"] == 0
        assert panel["opening_cluster_bootstrap"]["upper_bound_interval"] == [1, 1]
        assert panel["opening_cluster_bootstrap"]["replicates"] == 2000
        assert panel["opening_cluster_bootstrap"]["seed"] == 11300071
        assert panel["gate"]["required"] == (opponent in arena.PRIMARY_OPPONENTS)
    assert summary["by_opponent"]["full_afterstate"]["gate"]["passed"] is None
    assert summary["elo_estimate"] is None
    for opponent in arena.PRIMARY_OPPONENTS:
        wins = {o: 96 for o in arena.OPPONENTS}
        wins[opponent] = 57
        assert not arena.summarize(synthetic_outcomes(wins=wins))["gate_passed"]
    assert not arena.summarize(synthetic_outcomes(failures={"full_afterstate": 1}))["gate_passed"]
    mixed = arena.summarize(
        synthetic_outcomes(wins={o: 56 for o in arena.OPPONENTS}, draws={o: 4 for o in arena.OPPONENTS})
    )
    assert mixed["gate_passed"] and mixed["by_opponent"]["direct"]["delta"]["completed_points"] == 58


@pytest.mark.parametrize("fault", ["missing", "ordering", "arm", "opponent", "status", "result"])
def test_summary_rejects_incomplete_or_misidentified_results(fault):
    games = synthetic_outcomes()
    if fault == "missing":
        games.pop()
    elif fault == "ordering":
        games[0], games[1] = games[1], games[0]
    elif fault == "arm":
        games[0]["white_arm"] = "direct"
    elif fault == "opponent":
        games[0]["opponent"] = "action_only"
    elif fault == "status":
        games[0]["status"] = "unrecognized"
    else:
        games[-1]["result"] = "1/2-1/2"
    with pytest.raises(ValueError):
        arena.summarize(games)


class SyntheticModel(torch.nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.tensor(0.0))
        self.width, self.seed, self.depth, self.recurrence = 32, seed, 4, "residual"
        self.arm, self.branch_depth = arm, 2

    def choose(self, board, depth=None):
        assert depth == 4 and len(board.move_stack) >= 6
        assert torch.get_num_threads() == 2
        return response(board, {"white_arm": self.arm, "black_arm": self.arm, "seed": self.seed})


@pytest.fixture
def saved_arena(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "CandidateChess", SyntheticModel)
    monkeypatch.setattr(arena, "MAX_PLAYED_PLIES", 2)
    models = {(arm, seed): SyntheticModel(arm, seed) for arm in arena.ARMS for seed in arena.SEEDS}
    saved = tmp_path / "arena"
    previous = torch.get_num_threads()
    arena.run(saved, models, PLAN_HASH)
    yield saved
    torch.set_num_threads(previous)


def test_complete_saved_arena_roundtrip_and_no_overwrite(saved_arena):
    summary = arena.report(saved_arena, PLAN_HASH)
    assert summary["games"] == 288 and summary["status_counts"]["unfinished"] == 288
    assert summary["policy_calls"] == summary["played_plies"] == 576
    assert summary["aggregate"]["delta"]["score_lower_bound"] == 0
    assert summary["aggregate"]["delta"]["score_upper_bound"] == 1
    assert not summary["gate_passed"]
    assert arena.audit(saved_arena, PLAN_HASH) == summary
    models = {(a, s): SyntheticModel(a, s) for a in arena.ARMS for s in arena.SEEDS}
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
        game["white"] = "direct-97"
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
        (saved_arena / "game-288.json").unlink()
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
    monkeypatch.setattr(arena, "CandidateChess", SyntheticModel)
    models = {(a, s): SyntheticModel(a, s) for a in arena.ARMS for s in arena.SEEDS}
    models["direct", 97].recurrence = "anchor"
    with pytest.raises(ValueError):
        arena.run(tmp_path / "bad", models, PLAN_HASH)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("arm", "delta"), ("seed", 17), ("width", 128), ("depth", 8), ("branch_depth", 4)],
)
def test_model_metadata_is_checked_before_any_game(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(arena, "CandidateChess", SyntheticModel)
    models = {(a, s): SyntheticModel(a, s) for a in arena.ARMS for s in arena.SEEDS}
    setattr(models["direct", 97], field, value)
    with pytest.raises(ValueError, match="identity"):
        arena.run(tmp_path / "bad", models, PLAN_HASH)
    assert not (tmp_path / "bad").exists()


def test_model_manifest_includes_exact_tensor_digest_and_all_twelve_models(monkeypatch):
    monkeypatch.setattr(arena, "CandidateChess", SyntheticModel)
    models = {(a, s): SyntheticModel(a, s) for a in arena.ARMS for s in arena.SEEDS}
    before = arena._model_bindings(models)
    assert set(before) == {f"{a}-{s}" for a, s in models}
    assert before["delta-97"]["branch_depth"] == 2
    with torch.no_grad():
        models["delta", 97].placeholder.add_(1)
    after = arena._model_bindings(models)
    assert before["delta-97"]["state_sha256"] != after["delta-97"]["state_sha256"]
    assert before["direct-97"] == after["direct-97"]
    models.pop(("direct", 97))
    with pytest.raises(ValueError, match="twelve"):
        arena._model_bindings(models)


@pytest.mark.parametrize("fault", ["negative_time", "nonfinite_time", "failed", "model_metadata", "hash"])
def test_invalid_completion_or_model_manifest_is_rejected(saved_arena, fault):
    path = saved_arena / "completed.json"
    receipt = arena._read(path)
    if fault == "negative_time":
        receipt["wall_seconds"] = -1
    elif fault == "nonfinite_time":
        receipt["wall_seconds"] = float("inf")
    elif fault == "failed":
        receipt["status"] = "failed"
    else:
        started = arena._read(saved_arena / "started.json")
        if fault == "model_metadata":
            started["models"]["delta-97"]["branch_depth"] = 1
        else:
            started["models"]["delta-97"]["state_sha256"] = "not-a-digest"
        (saved_arena / "started.json").write_text(json.dumps(started))
        receipt["files"]["started.json"] = arena._hash(saved_arena / "started.json")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        arena.report(saved_arena, PLAN_HASH)


def test_unexpected_execution_failure_is_preserved_and_cannot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "CandidateChess", SyntheticModel)
    models = {(a, s): SyntheticModel(a, s) for a in arena.ARMS for s in arena.SEEDS}
    out = tmp_path / "interrupted"
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("synthetic driver failure")

    monkeypatch.setattr(arena, "play", fail)
    with pytest.raises(RuntimeError, match="synthetic driver"):
        arena.run(out, models, PLAN_HASH)
    failure = arena._read(out / "failed.json")
    assert len(calls) == 1 and failure["status"] == "failed"
    assert failure["plan_sha256"] == PLAN_HASH
    assert set(failure["files"]) == {"started.json"}
    assert failure["files"]["started.json"] == arena._hash(out / "started.json")
    assert not (out / "completed.json").exists()
    with pytest.raises(FileExistsError):
        arena.run(out, models, PLAN_HASH)
    assert len(calls) == 1


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_invalid_monotonic_times_are_not_game_results(elapsed):
    spec = arena.schedule()[0]
    ticks = iter((0.0, elapsed))
    policy = lambda board: response(board, spec)
    with pytest.raises(ValueError, match="monotonic"):
        arena.play(spec, policy, policy, timer=lambda: next(ticks))
