"""Prospective candidate-study data with root-and-successor exposure separation.

Prior inputs, recorded successors, arena positions and every legal successor of
the reused training roots are excluded. Each fresh accepted root and all of its
legal successors avoid that prior set and the other split's accepted exposures.
Within a split, successor overlap with other successors or roots is allowed;
only root/mirror duplicates are forbidden. This permits consecutive rollout
positions without claiming that they are independent observations.
"""

import copy
import hashlib
import math
import random
import time
from pathlib import Path

import chess
import chess.engine

from openjev.research import chess_anchor_data as source
from openjev.research import chess_capacity_arena as capacity_arena
from openjev.research import chess_capacity_data as capacity
from openjev.research import chess_spatial_data as native

TRAIN = capacity.TRAIN
ORIGINAL_COUNT = 32768
CAPACITY_EXECUTION = "runs/chess-capacity-v1/execution"
CAPACITY_PLAN = "evidence/chess-capacity-v1/plan.json"
CAPACITY_CONFIG = copy.deepcopy(capacity.CONFIG)
ADMISSION = {
    "version": "root-and-all-legal-successors-v1",
    "prior": "Accepted root and every legal successor avoid all prior natural and mirrored states",
    "across_splits": "Accepted root-plus-successor exposure sets are disjoint across splits",
    "within_split": "Successors may repeat or overlap other accepted roots; accepted roots remain globally mirror-unique",
    "state_identity": "Literal-EP first four FEN fields, ignoring move counters; reserve both color/rank mirrors",
}
CONFIG = {
    "teacher_nodes": 2000,
    "value_cp_scale": 600.0,
    "mate_cp": 10000,
    "teacher_threads": 1,
    "teacher_hash_mb": 16,
    "aux_actions": 4,
    "admission": ADMISSION,
    "splits": [
        {
            "name": "dev",
            "examples": 2048,
            "game_cap": 600,
            "max_plies": 64,
            "random_move_probability": 0.5,
            "seed_base": 111000000,
        },
        {
            "name": "shift",
            "examples": 2048,
            "game_cap": 600,
            "max_plies": 96,
            "random_move_probability": 0.1,
            "seed_base": 112000000,
        },
    ],
}


def _training(root):
    root = Path(root)
    receipt, members = capacity._bound_data((root / TRAIN).parent, ("train", "dev", "shift"))
    if receipt.get("counts", {}).get("train") != ORIGINAL_COUNT:
        raise ValueError("Original training count differs from the fixed study")
    rows = source._rows(members["train.jsonl"], ORIGINAL_COUNT)
    seen = set()
    for index, row in enumerate(rows):
        if (
            row.get("split") != "train"
            or row.get("id") != f"train-{index:05d}"
            or type(row.get("game_id")) is not int
            or row["game_id"] < 0
        ):
            raise ValueError("Invalid original training membership")
        board = source._board(row["fen"])
        canonical = native.symmetry_key(board)
        if row.get("state_key") != native.state_key(board) or canonical in seen:
            raise ValueError("Invalid or duplicated original training state")
        seen.add(canonical)
        source._legal_move(board, row["target_uci"])
        value = row.get("target_value")
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not -1 <= value <= 1
            or type(row.get("score_cp")) is not int
            or (row.get("mate") is not None and type(row["mate"]) is not int)
            or not math.isclose(value, math.tanh(row["score_cp"] / 600), abs_tol=1e-12)
        ):
            raise ValueError("Invalid original teacher target")
    return rows, members


def training_rows(root):
    """Return the original 32,768 rows, preserving every ID, value and label."""
    return _training(root)[0]


def _execution_members(root):
    directory = root / CAPACITY_EXECUTION
    raw = capacity._read(directory / "completed.json")
    complete = source._decode(raw)
    plan_raw = capacity._read(root / CAPACITY_PLAN)
    plan_hash = hashlib.sha256(plan_raw).hexdigest()
    if (
        complete.get("status") != "completed"
        or (directory / "failed.json").exists()
        or complete.get("plan_sha256") != plan_hash
        or not isinstance(complete.get("files"), dict)
    ):
        raise ValueError("Capacity execution did not complete against its bound plan")
    actual = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Capacity sources cannot contain symlinks")
        if path.is_file() and path != directory / "completed.json":
            actual[path.relative_to(directory).as_posix()] = native.sha(path)
    if actual != complete["files"]:
        raise ValueError("Capacity outer receipt does not bind all execution members")
    if source._decode(capacity._read(directory / "started.json")).get("plan_sha256") != plan_hash:
        raise ValueError("Capacity start and completion plan bindings differ")
    files = {f"{CAPACITY_EXECUTION}/{name}": digest for name, digest in actual.items()}
    files[f"{CAPACITY_EXECUTION}/completed.json"] = hashlib.sha256(raw).hexdigest()
    files[CAPACITY_PLAN] = plan_hash
    return files, plan_hash


def collect_exclusions(root):
    """Bind all prior evidence and reserve all native children of training roots.

    The caller freezes this returned file manifest and the reused training SHA in
    its study plan. Generation then binds the exact supplied exclusion-state list.
    Only state identities and hashes leave this collector, never held-out labels.
    """
    root = Path(root)
    inherited = capacity.collect_exclusions(root)
    states, files, counts = (
        set(inherited["states"]),
        dict(inherited["files"]),
        copy.deepcopy(inherited["counts"]),
    )
    by_file = counts["by_file"]
    mirror_classes = counts["unique_mirror_classes"]

    def bind(relative, digest, kind):
        if relative in files and files[relative] != digest:
            raise ValueError("Source changed during exclusion collection")
        files[relative] = digest
        by_file.setdefault(relative, {"kind": kind, "rows": 0, "positions": 0})

    def add(board):
        nonlocal mirror_classes
        key, mirror = native.state_key(board), native.state_key(board.mirror())
        if key not in states and mirror not in states:
            mirror_classes += 1
        states.add(key)

    execution_files, plan_hash = _execution_members(root)
    for relative, digest in execution_files.items():
        bind(relative, digest, "capacity_execution_member")
    directory = root / CAPACITY_EXECUTION
    expected = {s["name"]: s["examples"] for s in CAPACITY_CONFIG["splits"]}
    _, members = capacity._bound_data(directory / "data", expected, expected)
    if source._decode(members["started.json"]).get("config") != CAPACITY_CONFIG:
        raise ValueError("Capacity source configuration differs from its frozen contract")
    inputs = behaviors = auxiliaries = 0
    for split, count in expected.items():
        rows = source._rows(members[f"{split}.jsonl"], count)
        file_behavior = file_aux = 0
        for index, row in enumerate(rows):
            board = source._board(row["fen"])
            if (
                row.get("split") != split
                or row.get("id") != f"{split}-{index:05d}"
                or row.get("state_key") != native.state_key(board)
            ):
                raise ValueError("Capacity dataset input identity mismatch")
            add(board)
            if "behavior_uci" not in row or "next_fen" not in row:
                raise ValueError("Capacity behavior successor is missing")
            add(source._successor(board, {"uci": row["behavior_uci"], "next_fen": row["next_fen"]}))
            file_behavior += 1
            transitions = row.get("aux_transitions")
            if (
                not isinstance(transitions, list)
                or len(transitions) != min(CAPACITY_CONFIG["aux_actions"], board.legal_moves.count())
                or len({r["uci"] for r in transitions}) != len(transitions)
            ):
                raise ValueError("Capacity auxiliary transition membership mismatch")
            for transition in transitions:
                add(source._successor(board, transition))
                file_aux += 1
        by_file[f"{CAPACITY_EXECUTION}/data/{split}.jsonl"] = {
            "kind": "capacity_dataset",
            "rows": len(rows),
            "input_positions": len(rows),
            "behavior_successor_positions": file_behavior,
            "auxiliary_successor_positions": file_aux,
            "positions": len(rows) + file_behavior + file_aux,
        }
        inputs += len(rows)
        behaviors += file_behavior
        auxiliaries += file_aux
    # The frozen arena auditor verifies every prefix, accepted move, clock and
    # terminal status before any of its states enter this exclusion manifest.
    capacity_arena.report(directory / "arena", plan_hash)
    arena_positions = 0
    for spec in capacity_arena.schedule():
        relative = f"{CAPACITY_EXECUTION}/arena/{spec['id']}.json"
        game = source._decode(capacity._read(root / relative))
        board = source._board(game["initial_fen"])
        add(board)
        for move in game["moves"]:
            board.push(source._legal_move(board, move["move_uci"]))
            add(board)
        number = len(game["moves"]) + 1
        arena_positions += number
        by_file[relative] = {
            "kind": "capacity_arena_game",
            "rows": 1,
            "positions": number,
            "accepted_moves": len(game["moves"]),
        }
    training, training_members = _training(root)
    for name, raw in training_members.items():
        relative = ((root / TRAIN).parent / name).relative_to(root).as_posix()
        bind(relative, hashlib.sha256(raw).hexdigest(), "original_training_member")
    training_successors = 0
    before_keys, before_classes = len(states), mirror_classes
    for row in training:
        board = source._board(row["fen"])
        add(board)
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            add(child)
            training_successors += 1
    counts.update(
        {
            "source_files": len(files),
            "unique_state_keys": len(states),
            "unique_mirror_classes": mirror_classes,
            "observed_positions": counts["observed_positions"]
            + inputs
            + behaviors
            + auxiliaries
            + arena_positions
            + training_successors,
            "dataset_rows": counts["dataset_rows"] + inputs,
            "dataset_input_positions": counts["dataset_input_positions"] + inputs,
            "dataset_behavior_successor_positions": counts["dataset_behavior_successor_positions"]
            + behaviors,
            "dataset_auxiliary_successor_positions": counts["dataset_auxiliary_successor_positions"]
            + auxiliaries,
            "arena_games": counts.get("arena_games", 0) + len(capacity_arena.schedule()),
            "arena_positions": counts.get("arena_positions", 0) + arena_positions,
            "capacity_input_positions": inputs,
            "capacity_behavior_successor_positions": behaviors,
            "capacity_auxiliary_successor_positions": auxiliaries,
            "capacity_arena_positions": arena_positions,
            "training_candidate_successor_positions": training_successors,
            "training_successor_new_state_keys": len(states) - before_keys,
            "training_successor_new_mirror_classes": mirror_classes - before_classes,
            "reused_training_rows": len(training),
            "by_file": dict(sorted(by_file.items())),
        }
    )
    return {"states": sorted(states), "files": dict(sorted(files.items())), "counts": counts}


class _Admission:
    def __init__(self, states, config):
        self.exclusions = set(states)
        self.excluded_classes, self.prior = native._exclusion_keys(self.exclusions)
        self.root_reserved = set(self.prior)
        self.roots = set()
        self.exposures = {s["name"]: set() for s in config["splits"]}
        self.rejections = dict.fromkeys(
            ("prior_root", "duplicate_root", "cross_split_root", "prior_successor", "cross_split_successor"),
            0,
        )
        self.native_successors = self.accepted_successors = 0

    def consider(self, board, split):
        key, mirror = native.state_key(board), native.state_key(board.mirror())
        canonical = min(key, mirror)
        others = [value for name, value in self.exposures.items() if name != split]
        reason, transitions, exposures = "accepted", [], {key, mirror}
        if key in self.prior:
            reason = "prior_root"
        elif canonical in self.roots:
            reason = "duplicate_root"
        elif any(key in value for value in others):
            reason = "cross_split_root"
        else:
            prior_child = cross_child = False
            for move in sorted(board.legal_moves, key=lambda value: value.uci()):
                child = board.copy(stack=False)
                child.push(move)
                natural, reflected = native.state_key(child), native.state_key(child.mirror())
                self.native_successors += 1
                transitions.append({"uci": move.uci(), "next_fen": child.fen(en_passant="fen")})
                exposures.update((natural, reflected))
                prior_child |= natural in self.prior
                cross_child |= any(natural in value for value in others)
            if prior_child:
                reason = "prior_successor"
            elif cross_child:
                reason = "cross_split_successor"
        if reason != "accepted":
            self.rejections[reason] += 1
        return {
            "accepted": reason == "accepted",
            "reason": reason,
            "transitions": transitions,
            "exposures": exposures,
            "canonical": canonical,
            "root_pair": (key, mirror),
        }

    def commit(self, proposal, split):
        if not proposal["accepted"]:
            raise ValueError("Cannot admit a rejected position")
        self.roots.add(proposal["canonical"])
        self.root_reserved.update(proposal["root_pair"])
        self.exposures[split].update(proposal["exposures"])
        self.accepted_successors += len(proposal["transitions"])

    def snapshot(self):
        return {
            "rules": ADMISSION,
            "rejection_counts": self.rejections.copy(),
            "native_successor_evaluations": self.native_successors,
            "accepted_candidate_successors": self.accepted_successors,
            "exposure_state_keys_by_split": {name: len(value) for name, value in self.exposures.items()},
            "exposure_mirror_classes_by_split": {
                name: len(value) // 2 for name, value in self.exposures.items()
            },
        }


def _generate(out, labeler, config, states):
    """One bounded attempt; injectable teacher exists for synthetic tests only."""
    native._validate_config(config)
    if config.get("admission") != ADMISSION:
        raise ValueError("Candidate admission rules differ from their contract")
    admission = _Admission(states, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    native._write_new(out / "started.json", {"started_unix": started, "config": config})
    native._write_new(out / "excluded-states.json", sorted(admission.exclusions))
    counts = {s["name"]: 0 for s in config["splits"]}
    counters = dict.fromkeys(
        (
            "teacher_calls",
            "successful_teacher_calls",
            "requested_nodes",
            "reported_nodes",
            "known_node_calls",
            "generated_games",
        ),
        0,
    )
    counters.update(teacher_wall_seconds=0.0, teacher_call_wall_seconds=0.0)

    def receipt(status):
        return {
            "status": status,
            "started_unix": started,
            "completed_unix": time.time(),
            "counts": counts.copy(),
            "unique_states": len(admission.roots),
            "excluded_states": len(admission.exclusions),
            "excluded_symmetry_classes": len(admission.excluded_classes),
            "reserved_state_keys": len(admission.root_reserved),
            **counters,
            "unknown_node_calls": counters["teacher_calls"] - counters["known_node_calls"],
            "rollout_only_teacher_calls": counters["successful_teacher_calls"] - len(admission.roots),
            "teacher_cost_scope": "One call per visited nonterminal state, including every admission rejection",
            "auxiliary_cost_scope": "Four independent native auxiliary transitions plus every native legal candidate; no extra teacher labels",
            "admission": admission.snapshot(),
            "files": {
                p.name: native.sha(p)
                for p in sorted(out.iterdir())
                if p.name not in {"completed.json", "failed.json"}
            },
        }

    try:
        with (out / "games.jsonl").open("x") as games, (out / "analyses.jsonl").open("x") as analyses:
            offset = 0
            for split in config["splits"]:
                name = split["name"]
                with (out / f"{name}.jsonl").open("x") as rows:
                    for local_game in range(split["game_cap"]):
                        game_id, seed = offset + local_game, split["seed_base"] + local_game
                        board, rng = chess.Board(), random.Random(seed)
                        actions, sampled, stop = [], 0, "ply_cap"
                        counters["generated_games"] += 1
                        try:
                            for ply in range(split["max_plies"]):
                                if board.outcome(claim_draw=False) is not None:
                                    stop = "terminal"
                                    break
                                proposal = admission.consider(board, name)
                                fen, key = board.fen(en_passant="fen"), native.state_key(board)
                                label = native._call_teacher(
                                    labeler,
                                    board,
                                    config,
                                    analyses,
                                    counters,
                                    {
                                        "split": name,
                                        "game_id": game_id,
                                        "ply": ply,
                                        "fen": fen,
                                        "state_key": key,
                                        "labelled_position": proposal["accepted"],
                                        "excluded_position": key in admission.prior,
                                        "admission_reason": proposal["reason"],
                                        "candidate_successor_count": len(proposal["transitions"]),
                                    },
                                )
                                random_action = rng.random() < split["random_move_probability"]
                                legal = sorted(board.legal_moves, key=lambda move: move.uci())
                                action = (
                                    rng.choice(legal)
                                    if random_action
                                    else chess.Move.from_uci(label["target_uci"])
                                )
                                child = board.copy(stack=False)
                                child.push(action)
                                if proposal["accepted"]:
                                    row = {
                                        "id": f"{name}-{counts[name]:05d}",
                                        "split": name,
                                        "game_id": game_id,
                                        "game_seed": seed,
                                        "ply": ply,
                                        "fen": fen,
                                        "state_key": key,
                                        **{
                                            k: label[k]
                                            for k in ("target_uci", "target_value", "score_cp", "mate")
                                        },
                                        "behavior_uci": action.uci(),
                                        "next_fen": child.fen(en_passant="fen"),
                                        "aux_transitions": native.auxiliary_transitions(board, seed, ply),
                                        "candidate_transitions": proposal["transitions"],
                                    }
                                    native._append(rows, row)
                                    admission.commit(proposal, name)
                                    counts[name] += 1
                                    sampled += 1
                                actions.append(
                                    {"uci": action.uci(), "source": "random" if random_action else "teacher"}
                                )
                                board.push(action)
                                if counts[name] == split["examples"]:
                                    stop = "quota"
                                    break
                        except Exception:
                            stop = "failed"
                            raise
                        finally:
                            native._append(
                                games,
                                {
                                    "split": name,
                                    "game_id": game_id,
                                    "seed": seed,
                                    "sampled_positions": sampled,
                                    "actions": actions,
                                    "final_fen": board.fen(en_passant="fen"),
                                    "stop": stop,
                                },
                            )
                        if counts[name] == split["examples"]:
                            break
                if counts[name] != split["examples"]:
                    raise ValueError(f"{name} generated {counts[name]} positions before its fixed game cap")
                offset += split["game_cap"]
        result = receipt("completed")
        native._write_new(out / "completed.json", result)
        return result
    except Exception as exc:
        native._write_new(
            out / "failed.json", {**receipt("failed"), "error_type": type(exc).__name__, "error": str(exc)}
        )
        raise


def generate(out, engine, states):
    if Path(out).exists():
        raise FileExistsError(out)
    with chess.engine.SimpleEngine.popen_uci(str(engine)) as teacher:
        if not teacher.id.get("name", "").startswith("Stockfish 19"):
            raise ValueError("Expected the frozen Stockfish 19 teacher")
        teacher.configure({"Threads": CONFIG["teacher_threads"], "Hash": CONFIG["teacher_hash_mb"]})
        return _generate(out, lambda board: native.stockfish_label(teacher, board, CONFIG), CONFIG, states)


def validate(out, states):
    """Audit the base generator contract, then replay every admission decision."""
    out = Path(out)
    states = set(states)
    required = {"started.json", "excluded-states.json", "games.jsonl", "analyses.jsonl", "completed.json"}
    required.update(f"{split['name']}.jsonl" for split in CONFIG["splits"])
    if (
        out.is_symlink()
        or not out.is_dir()
        or {p.name for p in out.iterdir()} != required
        or any(p.is_symlink() or not p.is_file() for p in out.iterdir())
    ):
        raise ValueError("Fresh data requires the exact regular evidence members")
    rows = native.validate_data(out, CONFIG, states)
    admission = _Admission(states, CONFIG)
    games, calls = native._jsonl(out / "games.jsonl"), native._jsonl(out / "analyses.jsonl")
    lookup = {(r["game_id"], r["ply"]): r for values in rows.values() for r in values}
    expected_order, offset, call_index = [], 0, 0
    for split in CONFIG["splits"]:
        selected = [g for g in games if g["split"] == split["name"]]
        if not selected or [g["game_id"] for g in selected] != list(range(offset, offset + len(selected))):
            raise ValueError("Generation skipped or replaced scheduled games")
        expected_order.extend(selected)
        sampled = 0
        for game in selected:
            board = chess.Board()
            if sampled == split["examples"]:
                raise ValueError("Generation continued after the fixed quota")
            for ply, action in enumerate(game["actions"]):
                if sampled == split["examples"]:
                    raise ValueError("Game continued after the split quota")
                call = calls[call_index]
                call_index += 1
                if (call["game_id"], call["ply"]) != (game["game_id"], ply):
                    raise ValueError("Teacher calls are not in the fixed rollout order")
                proposal = admission.consider(board, split["name"])
                if (
                    call["labelled_position"] != proposal["accepted"]
                    or call.get("admission_reason") != proposal["reason"]
                    or call.get("candidate_successor_count") != len(proposal["transitions"])
                ):
                    raise ValueError("Recorded admission does not match prior and cross-split exposures")
                if proposal["accepted"]:
                    row = lookup[game["game_id"], ply]
                    if row.get("candidate_transitions") != proposal["transitions"]:
                        raise ValueError(
                            "Candidate transitions are incomplete, reordered or not exact native successors"
                        )
                    admission.commit(proposal, split["name"])
                    sampled += 1
                board.push_uci(action["uci"])
            expected_stop = (
                "quota"
                if sampled == split["examples"]
                else "ply_cap"
                if len(game["actions"]) == split["max_plies"]
                else "terminal"
                if board.outcome(claim_draw=False) is not None
                else None
            )
            if game["stop"] != expected_stop:
                raise ValueError("Game stopped outside its frozen quota, ply cap or terminal state")
        if sampled != split["examples"]:
            raise ValueError("Admission replay did not attain the frozen quota")
        offset += split["game_cap"]
    if expected_order != games or call_index != len(calls):
        raise ValueError("Split or teacher-call ordering mismatch")
    if source._decode(capacity._read(out / "completed.json")).get("admission") != admission.snapshot():
        raise ValueError("Admission receipt counts differ from replay")
    return rows
