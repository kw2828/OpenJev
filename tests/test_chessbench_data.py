"""Synthetic ChessBench-format tests; never load external benchmark records."""

import copy
import hashlib
import json
import os
import struct

import chess
import pytest

from openjev.research import chessbench_data as bc
from openjev.research.chess_spatial_data import state_key, symmetry_key


def varint(value):
    groups = []
    while value >= 128:
        groups.append((value & 127) | 128)
        value >>= 7
    return bytes(groups + [value])


def record(fen=chess.STARTING_FEN, move="e2e4"):
    raw = fen.encode("utf-8")
    return varint(len(raw)) + raw + move.encode("utf-8")


def bag(records):
    limits, offset = [], 0
    for value in records:
        offset += len(value)
        limits.append(offset)
    return b"".join(records) + b"".join(struct.pack("<Q", value) for value in limits)


def write_bag(tmp_path, contents):
    path = tmp_path / "synthetic.bag"
    path.write_bytes(contents)
    return path


def row(board, source_index, move=None):
    return {
        "source_index": source_index, "fen": board.fen(en_passant="fen"),
        "target_uci": move or next(iter(board.legal_moves)).uci(),
        "state_key": state_key(board), "symmetry_key": symmetry_key(board),
    }


def sequence_rows():
    board, rows = chess.Board(), []
    for index, uci in enumerate(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")):
        rows.append(row(board, index, uci))
        board.push_uci(uci)
    return rows


def test_known_beam_tuple_wire_layout_has_only_one_length_prefix(tmp_path):
    # Manually specified bytes, independent of the fixture writer's varint logic.
    payload = b"\x38rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1e2e4"
    assert payload == record()
    path = write_bag(tmp_path, payload + struct.pack("<Q", len(payload)))
    assert bc.decode_bc_bag(path) == [row(chess.Board(), 0, "e2e4")]


def test_embedded_final_limit_is_index_start_not_an_extra_record(tmp_path):
    board = chess.Board()
    board.push_uci("e2e4")
    payload = bag([record(), record(board.fen(en_passant="fen"), "e7e5")])
    path = write_bag(tmp_path, payload)
    before = path.stat()
    digest = hashlib.sha256(payload).hexdigest()
    path.chmod(0o444)
    decoded = bc.decode_bc_bag(path)
    assert decoded == [row(chess.Board(), 0, "e2e4"), row(board, 1, "e7e5")]
    assert all(set(value) == bc.ROW_FIELDS for value in decoded)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize(("fen", "uci"), [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q"),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n"),
    ("7k/8/8/3pP3/8/8/8/7K w - d6 0 2", "e5d6"),
])
def test_native_legal_special_moves_round_trip(tmp_path, fen, uci):
    path = write_bag(tmp_path, bag([record(fen, uci)]))
    assert bc.decode_bc_bag(path) == [row(chess.Board(fen), 0, uci)]


@pytest.mark.parametrize("fen_length", [127, 128, 129])
def test_varint_boundary_with_valid_normalized_fen(tmp_path, fen_length):
    prefix = chess.STARTING_FEN.rsplit(" ", 1)[0] + " "
    fen = prefix + "1" * (fen_length - len(prefix))
    assert len(fen) == fen_length
    path = write_bag(tmp_path, bag([record(fen, "e2e4")]))
    assert bc.decode_bc_bag(path)[0]["fen"] == fen


@pytest.mark.parametrize("value", [0, 1, 127, 128, 16383, 16384, (1 << 63) - 1])
def test_varint_nonnegative_signed_int64_boundaries(value):
    encoded = varint(value)
    assert bc._length(encoded + b"not part of the length") == (value, len(encoded))


@pytest.mark.parametrize("encoded", [b"", b"\x80", b"\x80\x00", b"\x81\x00",
                                      b"\xff" * 9 + b"\x01", b"\x80" * 10 + b"\x00"])
def test_varint_rejects_truncated_nonminimal_negative_or_overlong(encoded):
    with pytest.raises(ValueError, match="varint|signed-int64"):
        bc._length(encoded)


def test_empty_bag_is_zero_rows_and_selection_reports_shortfall(tmp_path):
    path = write_bag(tmp_path, b"")
    assert bc.decode_bc_bag(path) == []
    selection = bc.select([], set())
    receipt = selection["selection_receipt"]
    assert receipt["status"] == "failed" and receipt["shortfall"] == 4096
    assert receipt["native_child_visits"] == receipt["examined_rows"] == 0
    assert selection["selected"] == selection["rejection_log"] == []


@pytest.mark.parametrize("contents", [
    b"x", b"x" * 7, struct.pack("<Q", 0), struct.pack("<Q", 8),
    b"x" * 8 + struct.pack("<Q", 99),
    b"x" * 8 + struct.pack("<Q", (1 << 64) - 1),
    b"x" * 9 + struct.pack("<Q", 8),
])
def test_malformed_footer_bounds_or_alignment(tmp_path, contents):
    with pytest.raises(ValueError, match="index"):
        bc.decode_bc_bag(write_bag(tmp_path, contents))


@pytest.mark.parametrize("limits", [
    (0, 20, 30), (10, 10, 30), (20, 10, 30), (10, 31, 30),
])
def test_record_limits_must_be_monotonic_positive_and_within_payload(tmp_path, limits):
    contents = b"x" * 30 + b"".join(struct.pack("<Q", value) for value in limits)
    with pytest.raises(ValueError, match="strictly increasing"):
        bc.decode_bc_bag(write_bag(tmp_path, contents))


@pytest.mark.parametrize("removed", [1, 2, 7, 8, 9, 16])
def test_truncated_valid_bag_never_returns_partial_records(tmp_path, removed):
    contents = bag([record(), record(move="d2d4")])
    with pytest.raises(ValueError):
        bc.decode_bc_bag(write_bag(tmp_path, contents[:-removed]))


@pytest.mark.parametrize("contents", [
    b"\x00e2e4", b"\x80", b"\x7fshort", b"\x38" + chess.STARTING_FEN.encode(),
    b"\x38" + chess.STARTING_FEN.encode() + b"\x04e2e4",
    record() + b"junk", b"\x28\xb5\x2f\xfdzstd-content-is-not-a-BC-tuple",
])
def test_malformed_tuple_or_extra_component_bytes_are_rejected(tmp_path, contents):
    with pytest.raises(ValueError, match="record 0"):
        bc.decode_bc_bag(write_bag(tmp_path, bag([contents])))


@pytest.mark.parametrize("contents", [
    b"\x38\xff" + chess.STARTING_FEN.encode()[1:] + b"e2e4",
    record()[:-1] + b"\xff",
])
def test_both_strings_require_strict_utf8(tmp_path, contents):
    with pytest.raises(ValueError, match="invalid UTF-8"):
        bc.decode_bc_bag(write_bag(tmp_path, bag([contents])))


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN + " ", chess.STARTING_FEN.replace(" w ", "  w "),
    chess.STARTING_FEN.replace(" w ", "\tw "),
    chess.STARTING_FEN.replace("KQkq", "QKqk"),
    chess.STARTING_FEN.replace(" 0 1", " 00 1"),
    chess.STARTING_FEN.replace(" 0 1", " 0 0"),
    chess.STARTING_FEN.replace(" 0 1", " -1 1"),
    " ".join(chess.STARTING_FEN.split()[:4]),
    "8/8/8/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
    chess.STARTING_FEN.replace(" - ", " e3 "),
])
def test_fen_must_be_normalized_and_native_valid(tmp_path, fen):
    with pytest.raises(ValueError, match="FEN"):
        bc.decode_bc_bag(write_bag(tmp_path, bag([record(fen)])))


@pytest.mark.parametrize("move", ["e4", "O-O", "E2E4", "e2e4\n", "0000", "a7a8k", "e9e4", "N@e4",
                                    "e2e5", "e2e2", "e2e4q", "a7a8q", "e1g1"])
def test_move_syntax_and_legality_are_not_repaired(tmp_path, move):
    with pytest.raises(ValueError, match="Move|move"):
        bc.decode_bc_bag(write_bag(tmp_path, bag([record(move=move)])))


def test_record_error_identifies_original_physical_index(tmp_path):
    path = write_bag(tmp_path, bag([record(), record(move="e2e5")]))
    with pytest.raises(ValueError, match="record 1"):
        bc.decode_bc_bag(path)


def test_compressed_extension_and_symlinks_are_not_accepted(tmp_path):
    path = tmp_path / "compressed.bagz"
    path.write_bytes(bag([record()]))
    with pytest.raises(ValueError, match="uncompressed"):
        bc.decode_bc_bag(path)
    link = tmp_path / "linked.bag"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        bc.decode_bc_bag(link)


def test_nonregular_fifo_is_rejected_before_a_blocking_open(tmp_path):
    path = tmp_path / "pipe.bag"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        bc.decode_bc_bag(path)


def test_file_change_during_read_is_rejected(tmp_path, monkeypatch):
    path = write_bag(tmp_path, bag([record()]))
    actual_decode = bc._decode_bc_record

    def mutate_after_read(contents, source_index):
        decoded = actual_decode(contents, source_index)
        with path.open("ab") as stream:
            stream.write(b"x")
        return decoded

    monkeypatch.setattr(bc, "_decode_bc_record", mutate_after_read)
    with pytest.raises(ValueError, match="changed while"):
        bc.decode_bc_bag(path)


def expected_order(rows, seed=11400001):
    def rank(row):
        material = json.dumps([seed, row["state_key"], row["source_index"]],
                              ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
        return hashlib.sha256(material).hexdigest(), row["source_index"]
    return [value["source_index"] for value in sorted(rows, key=rank)]


def test_selection_is_label_independent_order_independent_and_reproducible():
    rows = sequence_rows()
    before = copy.deepcopy(rows)
    result = bc.select(rows, set(), limit=3)
    changed = copy.deepcopy(rows)
    for value in changed:
        value["target_uci"] = "not even a move; never inspected by select"
    altered = bc.select(list(reversed(changed)), set(), limit=3)
    unlabeled = bc.select([{key: value for key, value in row.items() if key != "target_uci"}
                           for row in rows], set(), limit=3)
    assert result["selection_receipt"] == altered["selection_receipt"] == unlabeled["selection_receipt"]
    assert result["rejection_log"] == altered["rejection_log"] == unlabeled["rejection_log"]
    receipt = result["selection_receipt"]
    assert receipt["selected_indices"] == expected_order(rows)[:3]
    assert receipt["unvisited_admission_indices"] == expected_order(rows)[3:]
    assert receipt["status"] == "completed" and receipt["shortfall"] == 0
    assert receipt["examined_rows"] == receipt["selected_count"] == 3
    assert rows == before
    assert result == bc.select(rows, set(), limit=3)
    result["selected"][0]["fen"] = "mutating an output does not change inputs"
    assert rows == before


def test_selection_seed_affects_position_order_but_never_labels():
    rows = sequence_rows()
    result = bc.select(rows, set(), limit=len(rows), seed=11400002)
    assert result["selection_receipt"]["selected_indices"] == expected_order(rows, 11400002)
    assert expected_order(rows, 11400002) != expected_order(rows, 11400001)


@pytest.mark.parametrize("mirrored", [False, True])
def test_prior_root_and_its_mirror_are_excluded_without_child_construction(mirrored):
    board = chess.Board()
    prior = state_key(board.mirror() if mirrored else board)
    result = bc.select([row(board, 0)], {prior}, limit=1)
    receipt = result["selection_receipt"]
    assert receipt["status"] == "failed" and receipt["shortfall"] == 1
    assert receipt["rejection_counts"] == {
        "terminal_root": 0, "prior_root": 1, "duplicate_root": 0, "prior_successor": 0,
    }
    assert receipt["native_child_visits"] == 0
    assert result["rejection_log"][0]["reason"] == "prior_root"


@pytest.mark.parametrize("mirrored", [False, True])
def test_any_prior_legal_successor_is_excluded_even_when_target_differs(mirrored):
    board, child = chess.Board(), chess.Board()
    child.push_uci("a2a4")
    prior = state_key(child.mirror() if mirrored else child)
    result = bc.select([row(board, 0, "e2e4")], {prior}, limit=1)
    receipt = result["selection_receipt"]
    assert receipt["status"] == "failed"
    assert receipt["native_child_visits"] == receipt["native_pushes"] == receipt["native_board_copies"] == 20
    assert receipt["rejection_counts"] == {
        "terminal_root": 0, "prior_root": 0, "duplicate_root": 0, "prior_successor": 1,
    }
    rejected = result["rejection_log"][0]
    assert rejected["excluded_successors"] == [{"uci": "a2a4", "state_key": state_key(child),
                                                "symmetry_key": symmetry_key(child)}]


def test_special_move_successors_are_included_in_prior_exclusion():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    child = board.copy()
    child.push_uci("e1g1")
    result = bc.select([row(board, 0, "a1a2")], {state_key(child)}, limit=1)
    receipt = result["selection_receipt"]
    assert receipt["native_child_visits"] == board.legal_moves.count()
    assert result["rejection_log"][0]["excluded_successors"][0]["uci"] == "e1g1"


def test_root_deduplication_ignores_counters_and_color_rank_mirror():
    board = chess.Board()
    counter_variant = board.copy()
    counter_variant.halfmove_clock, counter_variant.fullmove_number = 17, 42
    rows = [row(board, 0), row(board.mirror(), 1), row(counter_variant, 2)]
    result = bc.select(rows, set(), limit=2)
    receipt = result["selection_receipt"]
    assert receipt["status"] == "failed" and receipt["selected_count"] == 1
    assert receipt["rejection_counts"]["duplicate_root"] == 2
    assert receipt["native_child_visits"] == 20
    selected = receipt["selected_indices"][0]
    assert selected == expected_order(rows)[0]
    assert all(value["representative_source_index"] == selected for value in result["rejection_log"])


def test_selected_root_child_overlap_within_panel_is_allowed():
    rows = sequence_rows()[:2]
    result = bc.select(rows, set(), limit=2)
    receipt = result["selection_receipt"]
    assert receipt["status"] == "completed"
    assert set(receipt["selected_indices"]) == {0, 1}
    assert receipt["native_child_visits"] == 40
    assert result["rejection_log"] == []


@pytest.mark.parametrize("fen", [
    "7k/8/8/8/8/8/8/K7 w - - 0 1",  # Insufficient material, legal moves still exist.
    "7k/8/8/8/8/8/8/KR6 w - - 150 90",  # Automatic 75-move draw, legal moves still exist.
])
def test_fen_available_automatic_terminal_roots_are_logged_without_children(tmp_path, fen):
    board = chess.Board(fen)
    assert board.is_valid() and board.legal_moves.count() > 0
    assert board.is_game_over(claim_draw=False)
    source_row = row(board, 0)
    # Legal target decoding remains a format check; terminal filtering is selection-only.
    path = write_bag(tmp_path, bag([record(fen, source_row["target_uci"])]))
    rows = bc.decode_bc_bag(path)
    result = bc.select(rows, {state_key(board)}, limit=1)
    receipt = result["selection_receipt"]
    assert receipt["status"] == "failed" and receipt["shortfall"] == 1
    assert receipt["native_child_visits"] == 0
    assert receipt["rejection_counts"] == {
        "terminal_root": 1, "prior_root": 0, "duplicate_root": 0, "prior_successor": 0,
    }
    assert result["rejection_log"][0]["reason"] == "terminal_root"
    assert "Repetition history is unavailable" in receipt["scope"]


def test_claimable_fifty_move_draw_is_not_an_automatic_terminal_filter():
    board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 100 70")
    assert board.can_claim_fifty_moves() and not board.is_game_over(claim_draw=False)
    result = bc.select([row(board, 0)], set(), limit=1)
    assert result["selection_receipt"]["status"] == "completed"


def test_every_examined_or_unvisited_row_is_accounted_and_logs_are_label_free():
    rows = sequence_rows()
    prior = {rows[0]["state_key"]}
    result = bc.select(rows, prior, limit=2)
    receipt, log = result["selection_receipt"], result["rejection_log"]
    selected, rejected, unvisited = (set(receipt["selected_indices"]),
                                   {r["source_index"] for r in log},
                                   set(receipt["unvisited_admission_indices"]))
    assert not selected & rejected and not selected & unvisited and not rejected & unvisited
    assert selected | rejected | unvisited == {r["source_index"] for r in rows}
    assert receipt["examined_rows"] == len(selected) + len(rejected)
    assert sum(receipt["rejection_counts"].values()) == len(rejected)
    assert all("target_uci" not in r for r in log)
    assert "target_uci" not in json.dumps(receipt)


@pytest.mark.parametrize(("limit", "seed"), [(0, 1), (-1, 1), (True, 1), (1.0, 1),
                                             (1, -1), (1, True), (1, 1.0), (1, 1 << 64)])
def test_invalid_budget_and_seed_are_rejected(limit, seed):
    with pytest.raises(ValueError, match="limit|seed"):
        bc.select([], set(), limit=limit, seed=seed)


@pytest.mark.parametrize("excluded", ["not a set", {"bad"}, {chess.STARTING_FEN}, {"8/8/8/8/8/8/8/8 w - -"}])
def test_exclusions_must_be_normalized_valid_state_keys(excluded):
    with pytest.raises((TypeError, ValueError), match="state|FEN"):
        bc.select([], excluded, limit=1)


@pytest.mark.parametrize(("key", "value"), [("source_index", -1), ("source_index", True),
                                             ("source_index", 0.0), ("fen", "bad"),
                                             ("state_key", "bad"), ("symmetry_key", "bad")])
def test_selection_revalidates_identity_without_trusting_provided_keys(key, value):
    value_row = row(chess.Board(), 0)
    value_row[key] = value
    with pytest.raises(ValueError, match="indices|FEN|identity"):
        bc.select([value_row], set(), limit=1)


def test_duplicate_physical_indices_are_rejected_even_with_distinct_states():
    rows = sequence_rows()[:2]
    rows[1]["source_index"] = rows[0]["source_index"]
    with pytest.raises(ValueError, match="unique"):
        bc.select(rows, set(), limit=1)
