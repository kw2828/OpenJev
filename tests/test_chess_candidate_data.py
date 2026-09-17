"""Synthetic-only coverage of candidate exposure exclusions and fixed budgets."""

import copy
import json
import math

import chess
import chess.engine
import pytest

from openjev.research import chess_candidate_data as data
from openjev.research import chess_capacity_arena as arena
from openjev.research import chess_spatial_data as native


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def tiny_config(source=None, examples=3):
    config = copy.deepcopy(data.CONFIG if source is None else source)
    for index, split in enumerate(config["splits"]):
        split.update(
            examples=examples,
            game_cap=12,
            max_plies=12,
            random_move_probability=1.0,
            seed_base=810000 + index * 100,
        )
    return config


def label(board):
    return {
        "target_uci": min(move.uci() for move in board.legal_moves),
        "score_cp": 100,
        "target_value": math.tanh(100 / 600),
        "mate": None,
        "requested_nodes": 2000,
        "reported_nodes": 2001,
        "wall_seconds": 0.001,
    }


def test_frozen_config_and_explicit_overlap_contract():
    assert data.TRAIN == "runs/chess-spatial-v1/execution/data/train.jsonl"
    assert data.ORIGINAL_COUNT == 32768
    assert {
        key: data.CONFIG[key]
        for key in (
            "teacher_nodes",
            "teacher_threads",
            "teacher_hash_mb",
            "value_cp_scale",
            "mate_cp",
            "aux_actions",
        )
    } == {
        "teacher_nodes": 2000,
        "teacher_threads": 1,
        "teacher_hash_mb": 16,
        "value_cp_scale": 600.0,
        "mate_cp": 10000,
        "aux_actions": 4,
    }
    assert [
        (
            s["name"],
            s["examples"],
            s["game_cap"],
            s["max_plies"],
            s["random_move_probability"],
            s["seed_base"],
        )
        for s in data.CONFIG["splits"]
    ] == [
        ("dev", 2048, 600, 64, 0.5, 111000000),
        ("shift", 2048, 600, 96, 0.1, 112000000),
    ]
    assert "Successors may repeat" in data.ADMISSION["within_split"]


def test_admission_excludes_prior_successors_and_their_mirrors():
    board = chess.Board()
    child = board.copy()
    child.push_uci("e2e4")
    for key in (native.state_key(child), native.state_key(child.mirror())):
        admission = data._Admission([key], tiny_config())
        proposal = admission.consider(board, "dev")
        assert not proposal["accepted"] and proposal["reason"] == "prior_successor"
        assert len(proposal["transitions"]) == 20
        assert native.state_key(board) not in admission.prior


def test_same_split_successor_root_overlap_allowed_but_cross_split_rejected():
    admission = data._Admission([], tiny_config())
    board = chess.Board()
    proposal = admission.consider(board, "dev")
    admission.commit(proposal, "dev")
    child = board.copy()
    child.push_uci("e2e4")
    assert native.state_key(child) in admission.exposures["dev"]
    assert admission.consider(child, "dev")["accepted"]
    shifted = admission.consider(child, "shift")
    assert shifted["reason"] == "cross_split_root" and not shifted["accepted"]
    assert admission.consider(board.mirror(), "dev")["reason"] == "duplicate_root"


def test_cross_split_successor_collision_rejected_even_when_root_is_new():
    admission = data._Admission([], tiny_config())
    board = chess.Board()
    child = board.copy()
    child.push_uci("e2e4")
    # A different split has exposed this successor but not the current root.
    admission.exposures["dev"].update((native.state_key(child), native.state_key(child.mirror())))
    proposal = admission.consider(board, "shift")
    assert proposal["reason"] == "cross_split_successor" and not proposal["accepted"]


def test_generate_validate_complete_candidate_membership_and_call_costs(tmp_path, monkeypatch):
    config = tiny_config()
    monkeypatch.setattr(data, "CONFIG", config)
    excluded = [native.state_key(chess.Board())]
    result = data._generate(tmp_path / "generated", label, config, excluded)
    rows = data.validate(tmp_path / "generated", excluded)
    assert result["counts"] == {"dev": 3, "shift": 3}
    calls = native._jsonl(tmp_path / "generated/analyses.jsonl")
    assert result["teacher_calls"] == len(calls) > 6
    assert result["requested_nodes"] == 2000 * len(calls)
    assert result["reported_nodes"] == 2001 * len(calls)
    assert result["teacher_wall_seconds"] == pytest.approx(0.001 * len(calls))
    assert result["rollout_only_teacher_calls"] == len(calls) - 6
    exposures, roots = {}, set()
    prior, _ = native._exclusion_keys(excluded)
    for split, records in rows.items():
        values = set()
        for row in records:
            board = chess.Board(row["fen"])
            canonical = native.symmetry_key(board)
            assert canonical not in roots
            roots.add(canonical)
            values.add(canonical)
            legal = sorted(move.uci() for move in board.legal_moves)
            assert [value["uci"] for value in row["candidate_transitions"]] == legal
            for transition in row["candidate_transitions"]:
                child = board.copy()
                child.push_uci(transition["uci"])
                assert transition["next_fen"] == child.fen(en_passant="fen")
                values.add(native.symmetry_key(child))
        assert not values & prior
        exposures[split] = values
    assert not exposures["dev"] & exposures["shift"]
    with pytest.raises(FileExistsError):
        data._generate(tmp_path / "generated", label, config, excluded)


def test_seed_reproducibility_includes_all_admission_decisions(tmp_path):
    config = tiny_config()
    for name in ("left", "right"):
        data._generate(tmp_path / name, label, config, [])
    for filename in ("dev.jsonl", "shift.jsonl", "games.jsonl", "excluded-states.json"):
        assert (tmp_path / "left" / filename).read_bytes() == (tmp_path / "right" / filename).read_bytes()
    left, right = (native._jsonl(tmp_path / name / "analyses.jsonl") for name in ("left", "right"))
    for a, b in zip(left, right, strict=True):
        a.pop("call_wall_seconds")
        b.pop("call_wall_seconds")
        assert a == b


@pytest.mark.parametrize("excluded_kind", ["root", "successor"])
def test_fixed_game_cap_failure_charges_every_rejected_visit_without_retry(tmp_path, excluded_kind):
    config = tiny_config(examples=2)
    config["splits"] = [config["splits"][0]]
    config["splits"][0].update(game_cap=2, max_plies=1)
    blocked = chess.Board()
    if excluded_kind == "successor":
        blocked.push_uci("e2e4")
    excluded = [native.state_key(blocked)]
    with pytest.raises(ValueError, match="fixed game cap"):
        data._generate(tmp_path / "failed", label, config, excluded)
    receipt = json.loads((tmp_path / "failed/failed.json").read_text())
    assert receipt["status"] == "failed" and receipt["counts"] == {"dev": 0}
    assert receipt["teacher_calls"] == receipt["generated_games"] == 2
    assert receipt["requested_nodes"] == 4000 and receipt["reported_nodes"] == 4002
    reason = "prior_root" if excluded_kind == "root" else "prior_successor"
    assert receipt["admission"]["rejection_counts"][reason] == 2
    assert receipt["admission"]["native_successor_evaluations"] == (0 if excluded_kind == "root" else 40)
    calls = native._jsonl(tmp_path / "failed/analyses.jsonl")
    assert all(call["excluded_position"] == (excluded_kind == "root") for call in calls)
    assert not (tmp_path / "failed/completed.json").exists()


def test_validation_accepts_one_shot_exclusions_and_rejects_unbound_members(tmp_path, monkeypatch):
    config = tiny_config(examples=1)
    monkeypatch.setattr(data, "CONFIG", config)
    excluded = [native.state_key(chess.Board())]
    out = tmp_path / "data"
    data._generate(out, label, config, excluded)
    assert set(data.validate(out, iter(excluded))) == {"dev", "shift"}
    (out / "unbound.json").write_text("{}")
    with pytest.raises(ValueError):
        data.validate(out, excluded)


def test_teacher_failure_keeps_attempt_cost_and_no_retries(tmp_path):
    calls = []

    def broken(_):
        calls.append(1)
        raise RuntimeError("synthetic teacher failure")

    with pytest.raises(RuntimeError):
        data._generate(tmp_path / "failed", broken, tiny_config(), [])
    receipt = json.loads((tmp_path / "failed/failed.json").read_text())
    assert len(calls) == receipt["teacher_calls"] == receipt["unknown_node_calls"] == 1
    assert receipt["requested_nodes"] == 2000 and receipt["successful_teacher_calls"] == 0
    assert native._jsonl(tmp_path / "failed/analyses.jsonl")[0]["status"] == "failed"


def rebind(directory, member):
    receipt = json.loads((directory / "completed.json").read_text())
    receipt["files"][member] = native.sha(directory / member)
    write(directory / "completed.json", receipt)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_successor",
        "reordered_successors",
        "bad_successor",
        "admission_reason",
        "call_order",
        "counts",
        "early_stop",
    ],
)
def test_rehashed_candidate_evidence_corruption_rejected(tmp_path, monkeypatch, fault):
    config = tiny_config()
    monkeypatch.setattr(data, "CONFIG", config)
    out = tmp_path / "data"
    data._generate(out, label, config, [])
    if fault in ("missing_successor", "reordered_successors", "bad_successor"):
        name = "dev.jsonl"
        rows = native._jsonl(out / name)
        transitions = rows[0]["candidate_transitions"]
        if fault == "missing_successor":
            transitions.pop()
        elif fault == "reordered_successors":
            transitions.reverse()
        else:
            transitions[0]["next_fen"] = rows[0]["fen"]
    elif fault in ("admission_reason", "call_order"):
        name = "analyses.jsonl"
        rows = native._jsonl(out / name)
        if fault == "admission_reason":
            rows[0]["admission_reason"] = "prior_root"
        else:
            rows[0], rows[1] = rows[1], rows[0]
    elif fault == "early_stop":
        name = "games.jsonl"
        rows = native._jsonl(out / name)
        rows[0]["stop"] = "terminal"
    else:
        receipt = json.loads((out / "completed.json").read_text())
        receipt["admission"]["native_successor_evaluations"] += 1
        write(out / "completed.json", receipt)
        with pytest.raises(ValueError):
            data.validate(out, [])
        return
    write_rows(out / name, rows)
    rebind(out, name)
    with pytest.raises(ValueError):
        data.validate(out, [])


def fixture_sources(tmp_path, monkeypatch):
    old_config = tiny_config(data.CAPACITY_CONFIG, examples=2)
    original = (tmp_path / data.TRAIN).parent
    native.generate_data(original, label, old_config, [])
    monkeypatch.setattr(data, "ORIGINAL_COUNT", 2)
    states = set()
    old_rows = native._jsonl(original / "train.jsonl")
    for row in old_rows:
        states.add(native.state_key(chess.Board(row["fen"])))
    relative = data.TRAIN
    inherited = {
        "states": sorted(states),
        "files": {relative: native.sha(tmp_path / relative)},
        "counts": {
            "source_files": 1,
            "observed_positions": 2,
            "unique_state_keys": len(states),
            "unique_mirror_classes": len(states),
            "dataset_rows": 2,
            "dataset_input_positions": 2,
            "dataset_behavior_successor_positions": 0,
            "dataset_auxiliary_successor_positions": 0,
            "by_file": {relative: {"kind": "dataset", "rows": 2, "positions": 2}},
        },
    }
    monkeypatch.setattr(data.capacity, "collect_exclusions", lambda _: copy.deepcopy(inherited))
    config = tiny_config(data.CAPACITY_CONFIG, examples=2)
    for index, split in enumerate(config["splits"]):
        split["seed_base"] = 850000 + index * 100
    monkeypatch.setattr(data, "CAPACITY_CONFIG", config)
    execution = tmp_path / data.CAPACITY_EXECUTION
    native.generate_data(execution / "data", label, config, states)
    plan = tmp_path / data.CAPACITY_PLAN
    write(plan, {"fixture": "capacity-plan"})
    plan_hash = native.sha(plan)
    write(execution / "started.json", {"plan_sha256": plan_hash})
    # Full fixed96-game membership, with only prescribed prefix moves in this fixture.
    monkeypatch.setattr(arena, "MAX_PLAYED_PLIES", 0)
    directory = execution / "arena"
    directory.mkdir()
    bindings = {
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
    write(
        directory / "started.json",
        {"status": "started", "plan_sha256": plan_hash, "protocol": arena.protocol(), "models": bindings},
    )
    games = []
    for spec in arena.schedule():
        game = arena.play(spec, None, None)
        write(directory / f"{spec['id']}.json", game)
        (directory / f"{spec['id']}.pgn").write_text(arena.to_pgn(game))
        games.append(game)
    write(directory / "summary.json", arena.summarize(games))
    write(
        directory / "completed.json",
        {
            "status": "completed",
            "plan_sha256": plan_hash,
            "wall_seconds": 0.0,
            "files": {p.name: native.sha(p) for p in sorted(directory.iterdir())},
        },
    )
    bind_outer(execution, plan_hash)
    return tmp_path, inherited


def bind_outer(execution, plan_hash):
    write(
        execution / "completed.json",
        {
            "status": "completed",
            "plan_sha256": plan_hash,
            "files": {
                p.relative_to(execution).as_posix(): native.sha(p)
                for p in sorted(execution.rglob("*"))
                if p.is_file() and p != execution / "completed.json"
            },
        },
    )


def test_collector_includes_all_training_children_capacity_targets_and_arena_history(tmp_path, monkeypatch):
    root, inherited = fixture_sources(tmp_path, monkeypatch)
    original_bytes = (root / data.TRAIN).read_bytes()
    result = data.collect_exclusions(root)
    assert set(inherited["states"]) <= set(result["states"])
    original = native._jsonl(root / data.TRAIN)
    assert data.training_rows(root) == original
    assert (root / data.TRAIN).read_bytes() == original_bytes
    children = 0
    for row in original:
        board = chess.Board(row["fen"])
        for move in board.legal_moves:
            child = board.copy()
            child.push(move)
            assert native.state_key(child) in result["states"]
            children += 1
    for split in ("train", "dev", "shift"):
        for row in native._jsonl(root / data.CAPACITY_EXECUTION / "data" / f"{split}.jsonl"):
            for fen in [row["fen"], row["next_fen"]] + [r["next_fen"] for r in row["aux_transitions"]]:
                assert native.state_key(chess.Board(fen)) in result["states"]
    for spec in arena.schedule():
        board = chess.Board()
        for uci in spec["opening_moves"]:
            board.push_uci(uci)
            assert native.state_key(board) in result["states"]
    assert result["counts"]["training_candidate_successor_positions"] == children
    assert result["counts"]["capacity_input_positions"] == 6
    assert result["counts"]["capacity_behavior_successor_positions"] == 6
    assert result["counts"]["capacity_auxiliary_successor_positions"] == 24
    assert result["counts"]["capacity_arena_positions"] == 96 * 7
    assert result["counts"]["source_files"] == len(result["files"]) == len(result["counts"]["by_file"])
    assert result["counts"]["unique_mirror_classes"] == len(
        {native.symmetry_key(chess.Board(key + " 0 1")) for key in result["states"]}
    )
    for relative, digest in result["files"].items():
        assert digest == native.sha(root / relative)
    assert "target_uci" not in json.dumps(result) and "target_value" not in json.dumps(result)


@pytest.mark.parametrize(
    "fault",
    [
        "outer_hash",
        "outer_plan",
        "failed",
        "missing_game",
        "missing_data",
        "bad_behavior",
        "bad_auxiliary",
        "symlink",
    ],
)
def test_prior_source_hash_membership_and_rehashed_transition_corruption_rejected(
    tmp_path, monkeypatch, fault
):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    execution = root / data.CAPACITY_EXECUTION
    member = execution / "data/dev.jsonl"
    if fault == "outer_hash":
        member.write_bytes(member.read_bytes() + b"\n")
    elif fault == "outer_plan":
        receipt = json.loads((execution / "completed.json").read_text())
        receipt["plan_sha256"] = "c" * 64
        write(execution / "completed.json", receipt)
    elif fault == "failed":
        write(execution / "failed.json", {"status": "failed"})
    elif fault == "missing_game":
        (execution / "arena/game-096.json").unlink()
    elif fault == "missing_data":
        member.unlink()
    elif fault == "symlink":
        alternate = root / "alternate.jsonl"
        alternate.write_bytes(member.read_bytes())
        member.unlink()
        member.symlink_to(alternate)
    else:
        rows = native._jsonl(member)
        if fault == "bad_behavior":
            rows[0]["next_fen"] = rows[0]["fen"]
        else:
            rows[0]["aux_transitions"][0]["uci"] = "a1a8"
        write_rows(member, rows)
        rebind(execution / "data", "dev.jsonl")
        bind_outer(execution, native.sha(root / data.CAPACITY_PLAN))
    with pytest.raises(ValueError):
        data.collect_exclusions(root)


def test_training_rejects_changed_labels_without_touching_source(tmp_path, monkeypatch):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    original = root / data.TRAIN
    rows = native._jsonl(original)
    rows[0]["target_uci"] = "a1a8"
    write_rows(original, rows)
    rebind(original.parent, "train.jsonl")
    with pytest.raises(ValueError):
        data.training_rows(root)


def test_stockfish_wrapper_uses_exact_teacher_contract_with_stub(tmp_path, monkeypatch):
    config = tiny_config(examples=1)
    monkeypatch.setattr(data, "CONFIG", config)

    class Engine:
        def __init__(self):
            self.id = {"name": "Stockfish 19 synthetic"}
            self.calls = []
            self.settings = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def configure(self, value):
            self.settings.append(value)

        def analyse(self, board, limit, info):
            assert not board.move_stack and limit.nodes == 2000 and info
            self.calls.append(board.fen(en_passant="fen"))
            return {
                "pv": [min(board.legal_moves, key=lambda move: move.uci())],
                "score": chess.engine.PovScore(chess.engine.Cp(100), board.turn),
                "nodes": 2001,
            }

    teacher = Engine()
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _: teacher)
    receipt = data.generate(tmp_path / "generated", "not-a-real-engine", [])
    data.validate(tmp_path / "generated", [])
    assert teacher.settings[0] == {"Threads": 1, "Hash": 16}
    assert teacher.settings.count({"Clear Hash": None}) == len(teacher.calls) == receipt["teacher_calls"]
