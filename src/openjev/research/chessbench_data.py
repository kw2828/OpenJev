"""Read-only, strict ChessBench behavioral-cloning bag adapter.

Format references, inspected without reading benchmark data or labels:
https://github.com/google-deepmind/searchless_chess/blob/90ae0e6b121673fc3079aaeffa047580bb600c0a/src/bagz.py
https://github.com/google-deepmind/searchless_chess/blob/90ae0e6b121673fc3079aaeffa047580bb600c0a/src/constants.py
https://github.com/apache/beam/blob/v2.56.0/sdks/python/apache_beam/coders/coder_impl.py
https://github.com/apache/beam/blob/v2.56.0/sdks/python/apache_beam/coders/slow_stream.py

The uncompressed container ends with cumulative little-endian uint64 record
limits. Its final limit also identifies the start of that index. A top-level
TupleCoder(StrUtf8Coder, StrUtf8Coder) record has a positive varint FEN byte
length, the FEN, then the move as the entire remaining byte string. The final
tuple component has no length prefix. This module implements only that schema;
it never imports Beam, unpickles objects, executes data, or decompresses bags.
"""

import hashlib
import json
import mmap
import os
import re
import stat
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import chess

from openjev.research.chess_spatial_data import state_key, symmetry_key

UPSTREAM_REVISION = "90ae0e6b121673fc3079aaeffa047580bb600c0a"
SELECTION_VERSION = "chessbench-bc-selection-v1"
ROW_FIELDS = frozenset({"source_index", "fen", "target_uci", "state_key", "symmetry_key"})
_UCI = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?", flags=re.ASCII)


def _length(record: bytes) -> tuple[int, int]:
    """Read one minimally encoded, nonnegative signed-int64 Beam varint."""
    value = 0
    for index in range(min(len(record), 10)):
        byte = record[index]
        value |= (byte & 127) << (7 * index)
        if value >= 1 << 63:
            raise ValueError("FEN length exceeds nonnegative signed-int64 range")
        if not byte & 128:
            if index and not byte & 127:
                raise ValueError("FEN length varint is not minimally encoded")
            return value, index + 1
    raise ValueError("Unterminated or overlong FEN length varint")


def _board(fen: str) -> chess.Board:
    if not isinstance(fen, str) or len(fen.split(" ")) != 6:
        raise ValueError("FEN must have six normalized space-separated fields")
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError("Malformed standard-chess FEN") from error
    if not board.is_valid() or board.fen(en_passant="fen") != fen:
        raise ValueError("FEN is not a valid normalized standard-chess position")
    return board


def _decode_bc_record(record: bytes, source_index: int) -> dict:
    length, prefix = _length(record)
    end = prefix + length
    if length == 0 or end >= len(record):
        raise ValueError("FEN length exceeds record or leaves an empty move")
    try:
        fen, uci = record[prefix:end].decode("utf-8"), record[end:].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Behavioral-cloning record contains invalid UTF-8") from error
    board = _board(fen)
    if _UCI.fullmatch(uci) is None:
        raise ValueError("Move is not normalized standard UCI")
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as error:
        raise ValueError("Move is not normalized standard UCI") from error
    if move not in board.legal_moves:
        raise ValueError("Target move is illegal in its FEN")
    return {
        "source_index": source_index, "fen": fen, "target_uci": uci,
        "state_key": state_key(board), "symmetry_key": symmetry_key(board),
    }


def decode_bc_bag(path: str | Path) -> list[dict]:
    """Decode one uncompressed, embedded-index .bag, failing on any bad record.

    Source indices are zero-based physical record indices. Empty files represent
    zero records, as in upstream BagFileReader. Empty records are invalid for BC.
    No partial decoded result escapes an error. FENs and moves are not repaired.
    The source is opened read-only and must be a regular file, not a symlink.
    """
    path = Path(path)
    if path.suffix != ".bag" or path.is_symlink():
        raise ValueError("Expected an uncompressed regular .bag file, not a symlink")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular .bag file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Expected a regular .bag file")
        size = before.st_size
        if size == 0:
            return []
        if size < 8:
            raise ValueError("Truncated bag index footer")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content:
            (index_start,) = struct.unpack_from("<Q", content, size - 8)
            if not 0 < index_start <= size - 8 or (size - index_start) % 8:
                raise ValueError("Bag index start or index byte length is invalid")
            limits = []
            previous = 0
            for offset in range(index_start, size, 8):
                (limit,) = struct.unpack_from("<Q", content, offset)
                if not previous < limit <= index_start:
                    raise ValueError("Bag record limits must be strictly increasing within payload")
                limits.append(limit)
                previous = limit
            if previous != index_start:
                raise ValueError("Final record limit differs from index start")
            decoded, start = [], 0
            for index, end in enumerate(limits):
                try:
                    decoded.append(_decode_bc_record(content[start:end], index))
                except ValueError as error:
                    raise ValueError(f"Invalid behavioral-cloning record {index}: {error}") from error
                start = end
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("Bag changed while being decoded")
    return decoded


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                                     sort_keys=True, allow_nan=False).encode("ascii")).hexdigest()


def _excluded_classes(excluded_states: Iterable[str]) -> set[str]:
    if isinstance(excluded_states, (str, bytes)):
        raise TypeError("Excluded states must be an iterable of four-field state keys")
    result = set()
    for key in excluded_states:
        if not isinstance(key, str) or len(key.split(" ")) != 4:
            raise ValueError("Excluded states must be normalized four-field state keys")
        board = _board(key + " 0 1")
        if state_key(board) != key:
            raise ValueError("Excluded state key is not normalized")
        result.add(symmetry_key(board))
    return result


def select(rows: Sequence[Mapping], excluded_states: Iterable[str], limit: int = 4096,
           seed: int = 11400001) -> dict:
    """Select roots and all legal successors unseen in prior states or mirrors.

    Input identities are checked against their FENs. A row may omit its target
    label entirely: selection never reads or validates target_uci. The decoder
    validates labels before this function is used for actual data. Ordering is
    SHA-256 of canonical JSON [seed, state_key, source_index], followed
    by source_index as the collision tie-breaker. Input list order is irrelevant.

    Automatically terminal roots are rejected using only FEN-available state;
    repetition history is unavailable and claimable draws are not claimed.
    Root/mirror duplicates of accepted rows are excluded. Each other candidate
    root and every legal successor must avoid all prior state/mirror classes.
    Within the selected panel, children may overlap other children or roots.
    Counters do not distinguish states. Native successors follow sorted UCI
    order; all children are checked, even after detecting an excluded child.
    Target labels and model predictions do not guide any admission decision.

    Stop admission at the requested quota, recording later indices as unvisited.
    Return {selected, selection_receipt, rejection_log}. A quota shortfall gives
    selection_receipt.status == "failed"; the partial rows are evidence only.
    No extra source, replacement budget, or silent smaller panel is permitted.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("Selection limit must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("Selection seed must be a nonnegative uint64 integer")
    excluded = _excluded_classes(excluded_states)
    ranked, indices, identities = [], set(), []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Selection rows must be mappings")
        index = row.get("source_index")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("Source indices must be unique nonnegative integers")
        indices.add(index)
        board = _board(row.get("fen"))
        natural, canonical = state_key(board), symmetry_key(board)
        if row.get("state_key") != natural or row.get("symmetry_key") != canonical:
            raise ValueError("Selection identity does not match its FEN")
        identity = {"source_index": index, "fen": row["fen"],
                    "state_key": natural, "symmetry_key": canonical}
        identities.append(identity)
        rank = _digest([seed, natural, index])
        ranked.append((rank, index, row, identity, board))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected, selected_indices, selected_identities, rejections, seen = [], [], [], [], {}
    counts = {"terminal_root": 0, "prior_root": 0, "duplicate_root": 0, "prior_successor": 0}
    child_visits, examined, unvisited = 0, 0, []
    for rank, index, row, identity, board in ranked:
        if len(selected) == limit:
            unvisited.append(index)
            continue
        examined += 1
        canonical = identity["symmetry_key"]
        reason, extra = None, {}
        if board.is_game_over(claim_draw=False):
            reason = "terminal_root"
        elif canonical in excluded:
            reason = "prior_root"
        elif canonical in seen:
            reason = "duplicate_root"
            extra["representative_source_index"] = seen[canonical]
        else:
            matches = []
            for move in sorted(board.legal_moves, key=lambda item: item.uci()):
                child = board.copy(stack=False)
                child.push(move)
                child_visits += 1
                if symmetry_key(child) in excluded:
                    matches.append({"uci": move.uci(), "state_key": state_key(child),
                                    "symmetry_key": symmetry_key(child)})
            if matches:
                reason = "prior_successor"
                extra["excluded_successors"] = matches
        if reason is not None:
            rejection = {**identity, "rank_sha256": rank, "reason": reason, **extra}
            rejections.append(rejection)
            counts[reason] += 1
        else:
            seen[canonical] = index
            selected.append(dict(row))
            selected_indices.append(index)
            selected_identities.append(identity)
    receipt = {
        "version": SELECTION_VERSION, "seed": seed, "limit": limit,
        "status": "completed" if len(selected) == limit else "failed",
        "error": None if len(selected) == limit else "Insufficient eligible rows for requested quota",
        "input_rows": len(ranked), "examined_rows": examined,
        "selected_count": len(selected), "shortfall": max(0, limit - len(selected)),
        "selected_indices": selected_indices, "rejection_counts": counts,
        "unvisited_admission_indices": unvisited,
        "native_child_visits": child_visits, "native_board_copies": child_visits,
        "native_pushes": child_visits,
        "input_identity_sha256": _digest(sorted(identities, key=lambda item: item["source_index"])),
        "excluded_classes": len(excluded), "excluded_classes_sha256": _digest(sorted(excluded)),
        "selected_identity_sha256": _digest(selected_identities),
        "rejection_log_sha256": _digest(rejections),
        "ordering": "Ascending SHA256 of canonical JSON [seed, state_key, source_index], then source_index",
        "scope": "Label-independent FEN-available automatic-terminal rejection and root/all-legal-successor prior exclusion with mirrors; accepted roots are mirror-unique. Repetition history is unavailable. Within-panel successor overlaps are allowed. No game separation asserted.",
    }
    return {"selected": selected, "selection_receipt": receipt, "rejection_log": rejections}
