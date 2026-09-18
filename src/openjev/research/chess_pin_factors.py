"""Absolute-pin witnesses for a proposed constraint-aware chess representation.

This preserves geometric attack semantics. A witness is not a legal-move mask:
checks, king moves, en passant and other constraints remain separate. Pin
features and factor-graph networks are established ideas, not a novelty claim.
"""
import chess

VERSION = 'absolute-pin-witness-input-v1'
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def validate(board, perspective):
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard board')
    if type(perspective) is not bool:
        raise ValueError('Expected a player color')


def witness(color, king, blocker, attacker, perspective):
    square = (lambda s: s) if perspective else chess.square_mirror
    return (int(color != perspective), square(king), square(blocker), square(attacker))


def pin_witnesses(board, perspective):
    """Sorted (owner,king,blocker,attacker) tuples in a fixed player frame.

Scan the eight rays from each king. An own first blocker and opposing matching
slider as the next occupied square define an absolute-pin witness. Does not
call the chess library's attack, pin or legal-move routines.
"""
    validate(board, perspective)
    result = []
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        for dx, dy in DIRECTIONS:
            file, rank = chess.square_file(king)+dx, chess.square_rank(king)+dy
            blocker = None
            while 0 <= file < 8 and 0 <= rank < 8:
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece is not None:
                    if blocker is None:
                        if piece.color != color or piece.piece_type == chess.KING:
                            break
                        blocker = square
                    else:
                        slider = chess.BISHOP if dx and dy else chess.ROOK
                        if piece.color != color and piece.piece_type in (slider, chess.QUEEN):
                            result.append(witness(color, king, blocker, square, perspective))
                        break
                file += dx; rank += dy
    return tuple(sorted(result))


def reference_witnesses(board, perspective):
    """Independent bitboard pin API and occupancy-removal attack query.

Shares python-chess board/rule primitives with the caller. It does not use the
primary ray scanner, its direction list, or the primary canonicalizer.
"""
    validate(board, perspective)
    result = []
    for square, piece in board.piece_map().items():
        color = piece.color
        if piece.piece_type == chess.KING or not board.is_pinned(color, square):
            continue
        king = board.king(color)
        before = set(board.attackers(not color, king))
        exposed = set(board.attackers(not color, king, occupied=board.occupied ^ chess.BB_SQUARES[square]))-before
        for attacker in sorted(exposed):
            if board.piece_type_at(attacker) not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                raise AssertionError('A pin exposed a non-slider')
            values = (king, square, attacker)
            if not perspective:
                values = tuple(chess.square(chess.square_file(s), 7-chess.square_rank(s)) for s in values)
            result.append((int(color != perspective), *values))
    return tuple(sorted(result))


def changes(before, after):
    """Append role 0=retained, 1=added, 2=removed to each square-identity tuple."""
    root, child = set(before), set(after)
    if len(root) != len(before) or len(child) != len(after):
        raise ValueError('Duplicate pin witness')
    return tuple((*row, 0 if row in root and row in child else 1 if row in child else 2)
                 for row in sorted(root | child))


def candidate_factors(board, *, reference=False):
    perspective = board.turn
    validate(board, perspective)
    if board.is_game_over(claim_draw=False):
        raise ValueError('Expected a nonterminal board')
    extract = reference_witnesses if reference else pin_witnesses
    before = extract(board, perspective)
    menu = tuple(sorted(move.uci() for move in board.legal_moves))
    records = []
    for uci in menu:
        child = board.copy(stack=True); child.push_uci(uci)
        after = extract(child, perspective)
        if reference:
            # Direct membership classification, without the primary set-union helper.
            factors = []
            for item in sorted(set(before+after)):
                present = item in before, item in after
                factors.append((*item, {(True, True): 0, (False, True): 1, (True, False): 2}[present]))
            factors = tuple(factors)
        else:
            factors = changes(before, after)
        records.append({'uci': uci, 'after': after, 'factors': factors})
    return {'before': before, 'candidates': records}
