"""Target-only ChessBench candidate-policy evaluation, with replayable logits.

This evaluator makes no model calls during audit. Legal menus and all prediction
arithmetic are reconstructed from saved finite logits and native chess rules.
Behavioral-cloning rows have no value labels; no value metrics are fabricated.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import chess
import torch

from openjev.research.chess_candidate import ARMS, CandidateChess, encode_batch
from openjev.research.chess_spatial_data import state_key, symmetry_key

VERSION = "chessbench-candidate-evaluation-v1"
ROW_FIELDS = frozenset({"source_index", "fen", "target_uci", "state_key", "symmetry_key"})
MODEL_FIELDS = frozenset({"arm", "seed", "width", "root_depth", "branch_depth", "state_sha256"})
PREDICTION_FIELDS = (
    ROW_FIELDS
    | MODEL_FIELDS
    | {
        "version",
        "legal_ids",
        "logits",
        "choice",
        "target_index",
        "correct",
        "target_nll",
        "target_probability",
        "confidence",
    }
)
TIMING_SCOPE = (
    "CPU batch board construction, native successor preparation, forward and prediction writing. "
    "Model loading, input validation, state hashing and output audit excluded. "
    "Batch wall time is not single-decision latency."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "Expected a finite number")
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        _require(
            tensor.device.type == "cpu" and tensor.dtype == torch.float32 and torch.isfinite(tensor).all(),
            "Evaluation requires finite CPU float32 model state",
        )
        metadata = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_sha256(model):
    """Hash the exact finite CPU-float32 parameter/buffer state without inference."""
    return _state_sha(model)


def input_rows_sha256(rows):
    """Hash input rows, including target identity, in their recorded panel order."""
    return _digest(list(rows))


def _board(row):
    _require(type(row) is dict and set(row) == ROW_FIELDS, "Unexpected ChessBench row schema")
    _require(type(row["source_index"]) is int and row["source_index"] >= 0, "Invalid source index")
    fen = row["fen"]
    _require(type(fen) is str and len(fen.split(" ")) == 6, "Expected a complete normalized FEN")
    board = chess.Board(fen)
    _require(
        board.is_valid()
        and not board.chess960
        and board.fen(en_passant="fen") == fen
        and not board.is_game_over(claim_draw=False),
        "Expected a valid nonterminal standard-chess board",
    )
    _require(
        row["state_key"] == state_key(board) and row["symmetry_key"] == symmetry_key(board),
        "Row identity differs from its FEN",
    )
    menu = sorted(move.uci() for move in board.legal_moves)
    _require(type(row["target_uci"]) is str and row["target_uci"] in menu, "Target is not a legal UCI move")
    return board


def _rows(rows):
    rows = list(rows)
    _require(bool(rows), "Evaluation requires at least one row")
    seen = set()
    for row in rows:
        _board(row)
        _require(row["source_index"] not in seen, "Duplicate source index")
        seen.add(row["source_index"])
    # A local immutable-by-convention copy prevents the evaluator attaching labels to boards.
    return [dict(row) for row in rows]


def _arithmetic(logits, legal_ids, target):
    _require(type(logits) is list and len(logits) == len(legal_ids), "Incomplete valid-logit menu")
    for logit in logits:
        _number(logit)
    best = max(range(len(logits)), key=logits.__getitem__)
    maximum = logits[best]
    log_normalizer = math.log(math.fsum(math.exp(value - maximum) for value in logits))
    target_index = legal_ids.index(target)
    nll = (maximum - logits[target_index]) + log_normalizer
    _number(nll)
    _require(nll >= 0, "Negative menu NLL")
    return {
        "choice": legal_ids[best],
        "target_index": target_index,
        "correct": legal_ids[best] == target,
        "target_nll": nll,
        "target_probability": math.exp(-nll),
        "confidence": math.exp(-log_normalizer),
    }


def _metrics(predictions):
    count = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    return {
        "positions": count,
        "correct": correct,
        "agreement": correct / count,
        "mean_nll": math.fsum(row["target_nll"] for row in predictions) / count,
        "mean_confidence": math.fsum(row["confidence"] for row in predictions) / count,
        "mean_target_probability": math.fsum(row["target_probability"] for row in predictions) / count,
    }


def _decode(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            _require(name not in result, "Duplicate JSON key")
            result[name] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _model_identity(identity):
    _require(
        identity["arm"] in ARMS
        and type(identity["seed"]) is int
        and type(identity["width"]) is int
        and identity["width"] > 0
        and identity["root_depth"] == 4
        and identity["branch_depth"] == 2,
        "Prediction model identity differs",
    )
    digest = identity["state_sha256"]
    _require(
        type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
        "Invalid model-state fingerprint",
    )


def audit_predictions(path, rows, expected_arm=None):
    """Return metrics after native-menu, coverage and float64 softmax auditing.

    This verifies saved arithmetic and identities, not whether a model produced
    particular logits. The evaluator's model-state and file-hash receipt supplies
    that binding. No neural forward, target value, or engine call is performed.
    """
    if expected_arm is not None:
        _require(expected_arm in ARMS, "Unknown expected arm")
    rows = _rows(rows)
    raw = Path(path).read_bytes()
    _require(raw.endswith(b"\n"), "Incomplete JSONL record")
    predictions = [_decode(line) for line in raw.splitlines()]
    _require(len(predictions) == len(rows), "Prediction row coverage differs")
    identity = None
    for prediction, row in zip(predictions, rows, strict=True):
        _require(
            type(prediction) is dict and set(prediction) == PREDICTION_FIELDS, "Prediction schema differs"
        )
        _require(prediction["version"] == VERSION, "Prediction version differs")
        _require(type(prediction["source_index"]) is int, "Invalid prediction source index")
        _require(
            {k: prediction[k] for k in ROW_FIELDS} == row, "Prediction source identity or row order differs"
        )
        current = {k: prediction[k] for k in MODEL_FIELDS}
        _model_identity(current)
        _require(expected_arm is None or current["arm"] == expected_arm, "Prediction arm differs")
        if identity is None:
            identity = current
        _require(identity == current, "Mixed model identities in one prediction panel")
        legal_ids = sorted(move.uci() for move in _board(row).legal_moves)
        _require(prediction["legal_ids"] == legal_ids, "Ordered native legal menu differs")
        calculated = _arithmetic(prediction["logits"], legal_ids, row["target_uci"])
        for key in ("choice", "target_index", "correct"):
            _require(
                type(prediction[key]) is type(calculated[key]) and prediction[key] == calculated[key],
                "Stable argmax or target identity differs",
            )
        for key in ("target_nll", "target_probability", "confidence"):
            actual = _number(prediction[key])
            _require(actual >= 0 and (key == "target_nll" or actual <= 1), "Prediction metric outside bounds")
            _require(
                math.isclose(actual, calculated[key], rel_tol=1e-12, abs_tol=1e-14),
                "Saved softmax arithmetic differs",
            )
    return _metrics(predictions)


def evaluate(model, rows, outpath, batch_size=16):
    """Write one exclusive CPU JSONL panel and return its completed cost receipt.

    Failures retain any partial output and are never retried or overwritten.
    Native successors are constructed afresh only for delta/full-afterstate.
    """
    _require(isinstance(model, CandidateChess), "Expected a CandidateChess model")
    _require(model.depth == 4 and model.branch_depth == 2, "Expected root depth four and branch depth two")
    _require(type(batch_size) is int and batch_size > 0, "Batch size must be a positive integer")
    rows = _rows(rows)
    identity = {
        "arm": model.arm,
        "seed": model.seed,
        "width": model.width,
        "root_depth": model.depth,
        "branch_depth": model.branch_depth,
        "state_sha256": _state_sha(model),
    }
    _model_identity(identity)
    path = Path(outpath)
    modes, previous_threads = (
        [(module, module.training) for module in model.modules()],
        torch.get_num_threads(),
    )
    predictions, legal_count, successor_count, batches = [], 0, 0, 0
    # Exclusive creation happens before any inference. Parent orchestration owns failure receipts.
    with path.open("x", encoding="utf-8") as stream:
        try:
            torch.set_num_threads(2)
            model.eval()
            started = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    batch_rows = rows[start : start + batch_size]
                    boards = [chess.Board(row["fen"]) for row in batch_rows]
                    inputs, menus = encode_batch(boards, model.arm, device="cpu")
                    logits, value, hidden = model(**inputs, depth=4)
                    _require(
                        logits.device.type == "cpu" and logits.shape == inputs["legal_mask"].shape,
                        "Model logits shape/device differs",
                    )
                    _require(
                        torch.isfinite(logits[inputs["legal_mask"]]).all()
                        and torch.isfinite(value).all()
                        and torch.isfinite(hidden).all(),
                        "Nonfinite model output",
                    )
                    count = sum(map(len, menus))
                    native = int(inputs["successors"].shape[0]) if "successors" in inputs else 0
                    _require(
                        native == (count if model.arm in ("delta", "full_afterstate") else 0),
                        "Native successor count differs",
                    )
                    legal_count += count
                    successor_count += native
                    batches += 1
                    for index, (row, menu) in enumerate(zip(batch_rows, menus, strict=True)):
                        raw_logits = logits[index, : len(menu)].double().tolist()
                        record = {
                            "version": VERSION,
                            **row,
                            **identity,
                            "legal_ids": list(menu),
                            "logits": raw_logits,
                            **_arithmetic(raw_logits, menu, row["target_uci"]),
                        }
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        predictions.append(record)
                stream.flush()
            elapsed = time.perf_counter() - started
        finally:
            for module, mode in modes:
                module.training = mode
            torch.set_num_threads(previous_threads)
    after = {
        "arm": model.arm,
        "seed": model.seed,
        "width": model.width,
        "root_depth": model.depth,
        "branch_depth": model.branch_depth,
        "state_sha256": _state_sha(model),
    }
    _require(after == identity, "Model state changed during evaluation")
    metrics = _metrics(predictions)
    _require(
        audit_predictions(path, rows, expected_arm=model.arm) == metrics, "Saved prediction audit differs"
    )
    return {
        "status": "completed",
        "version": VERSION,
        "model": identity,
        "metrics": metrics,
        "input_rows_sha256": input_rows_sha256(rows),
        "predictions_sha256": _file_sha(path),
        "model_state_unchanged": True,
        "device": "cpu",
        "torch_threads": 2,
        "batch_size": batch_size,
        "batches": batches,
        "evaluation_wall_seconds": elapsed,
        "timing_scope": TIMING_SCOPE,
        "root_encodings": len(rows),
        "candidate_evaluations": legal_count,
        "native_successors": successor_count,
        "native_board_copies": successor_count,
        "native_pushes": successor_count,
        "successor_encodings": successor_count,
        "probability_semantics": "Uncalibrated softmax over the complete native legal menu; not absolute move quality.",
    }
