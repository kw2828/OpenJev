"""Standard-chess candidate adapter and synchronous, auditable 5+0 game runner.

Policies receive an isolated ``chess.Board`` and return a JSON-compatible mapping
with ``choice`` equal to a legal UCI move ID. Optional ``probabilities`` must cover
every legal move. Candidate preparation, inference and response parsing performed
inside the policy all consume its clock. A policy must synchronize accelerator
work before returning. Calls are timed on return, not forcibly interrupted; a
remote policy must set its own transport deadline. No retry or fallback is used.

``moves`` contains only played moves; ``attempts`` additionally retains discarded
timeout/invalid responses. Invalid responses and policy exceptions are failed,
unscored games (``*``). A ply cap is unfinished, never a draw. Claimable draws are
ignored by default; ``claim_draw=True`` explicitly auto-claims them for both sides.
Flag fall loses unless python-chess recognizes the opponent's material as unable
to mate. This uses its material-based test, not a general dead-position solver.
"""

import json
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import chess
import chess.pgn

MAX_LEGAL_CANDIDATES = 218
Policy = Callable[[chess.Board], Mapping[str, Any]]
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _standard_board(board: chess.Board) -> None:
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard-chess Board')


def chess_record(board: chess.Board, history_plies: int | None = 16) -> dict[str, Any]:
    """Serialize observable state and all legal candidates, ordered by UCI ID.

    ``None`` includes the complete available move stack; zero omits moves. A FEN
    alone does not reconstruct earlier history. Terminal positions can have zero
    candidates. This dictionary bypasses the application API's twelve-choice cap;
    scorers must explicitly support 1..218 candidates without truncating the menu.
    No engine evaluation, target move, or policy-private board attribute is read.
    """
    _standard_board(board)
    if history_plies is not None and (type(history_plies) is not int or history_plies < 0):
        raise ValueError('history_plies must be a nonnegative integer or None')
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    if len(legal) > MAX_LEGAL_CANDIDATES:
        raise ValueError('Legal move count exceeds the standard-chess candidate bound')
    available = len(board.move_stack)
    history = board.move_stack if history_plies is None else board.move_stack[max(0, available-history_plies):]
    side = 'white' if board.turn else 'black'
    board_rows = str(board).splitlines()
    ascii_board = '\n'.join(f'{8-i} {row}' for i, row in enumerate(board_rows)) + '\n  a b c d e f g h'
    context = (
        f'Standard chess. Side to move: {side}.\nFEN: {board.fen()}\n'
        f'Board (uppercase white, lowercase black):\n{ascii_board}\n'
        f'Recent UCI history ({len(history)} of {available} available plies): '
        + (' '.join(move.uci() for move in history) or '(none)')
    )
    return {
        'context': context,
        'question': f'Which legal move should {side} play? Select one candidate ID.',
        'candidates': [{
            'id': move.uci(),
            'description': (
                f'{board.san(move)}; {chess.square_name(move.from_square)} to '
                f'{chess.square_name(move.to_square)}; UCI {move.uci()}'
            ),
        } for move in legal],
    }


def random_policy(seed: int) -> Policy:
    """Return a reproducible uniform legal-move baseline with its own RNG."""
    rng = random.Random(seed)

    def choose(board: chess.Board) -> dict[str, Any]:
        legal = sorted(move.uci() for move in board.legal_moves)
        if not legal:
            raise ValueError('Random policy called on a position without legal moves')
        return {
            'choice': rng.choice(legal),
            'probabilities': {move: 1/len(legal) for move in legal},
            'policy_kind': 'seeded_uniform_random', 'seed': seed,
        }

    return choose


def greedy_material_policy() -> Policy:
    """One-ply material heuristic with deterministic UCI tie-breaking, no search.

    Scores the resulting material balance with P/N/B/R/Q = 1/3/3/5/9. It has no
    learned weights, positional evaluation, checkmate bonus or opponent lookahead.
    Promotion and en passant are accounted for by applying native legal moves.
    """
    def choose(board: chess.Board) -> dict[str, Any]:
        color = board.turn
        scored = []
        for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
            after = board.copy(stack=False)
            after.push(move)
            value = sum(
                weight*(len(after.pieces(piece, color))-len(after.pieces(piece, not color)))
                for piece, weight in PIECE_VALUES.items()
            )
            scored.append((move.uci(), value))
        if not scored:
            raise ValueError('Material policy called on a position without legal moves')
        choice, value = max(scored, key=lambda item: item[1])
        return {'choice': choice, 'policy_kind': 'one_ply_material_heuristic', 'material_balance': value}

    return choose


@dataclass
class GameResult:
    initial_fen: str
    white: str
    black: str
    initial_clocks: dict[str, float]
    clocks: dict[str, float]
    claim_draw: bool
    max_plies: int
    moves: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    result: str = '*'
    termination: str = 'max_plies'
    status: str = 'unfinished'
    final_fen: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_pgn(self) -> str:
        """Export only actual moves, with starting FEN, result and clock comments."""
        game = chess.pgn.Game()
        game.setup(chess.Board(self.initial_fen))
        game.headers.update({
            'Event': 'OpenJev chess prototype', 'White': self.white, 'Black': self.black,
            'Result': self.result, 'TimeControl': f"{self.initial_clocks['white']:g}+0",
            'Termination': ('time forfeit' if self.termination == 'timeout' else
                            'normal' if self.status == 'completed' else 'unterminated'),
            'OpenJevTermination': self.termination,
            'OpenJevStatus': self.status,
            'DrawClaims': 'automatic claims enabled' if self.claim_draw else 'automatic draws only',
        })
        node = game
        for row in self.moves:
            node = node.add_variation(chess.Move.from_uci(row['move_uci']))
            node.set_clock(row['clocks'][row['side']])
            node.set_emt(row['latency_ms']/1000)
        return game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True))


def _response(response: Any, legal_ids: set[str]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError('Policy must return a mapping')
    # Deep snapshot prevents later mutation and ensures the trace can be saved.
    value = json.loads(json.dumps(dict(response), allow_nan=False))
    if not isinstance(value.get('choice'), str):
        raise TypeError('Policy choice must be a UCI string')
    if 'probabilities' in value:
        probabilities = value['probabilities']
        if not isinstance(probabilities, dict) or set(probabilities) != legal_ids:
            raise ValueError('Probabilities must cover exactly all legal move IDs')
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0
               for p in probabilities.values()):
            raise ValueError('Probabilities must be finite and nonnegative')
        if not math.isclose(sum(probabilities.values()), 1., rel_tol=0., abs_tol=1e-6):
            raise ValueError('Probabilities must sum to one')
    return value


def play_game(
    white: Policy,
    black: Policy,
    *,
    white_name: str = 'white',
    black_name: str = 'black',
    initial_fen: str = chess.STARTING_FEN,
    clock_seconds: float = 300.,
    max_plies: int = 256,
    claim_draw: bool = False,
    timer: Callable[[], float] | None = None,
) -> GameResult:
    """Play one game with no increment, retries, hidden moves or fallback.

    The injected ``timer`` exists for deterministic clock tests; production uses
    ``time.perf_counter``. Clock expiry takes precedence over an invalid response
    or policy exception, and any proposed move after expiry is discarded.
    """
    if not math.isfinite(clock_seconds) or clock_seconds <= 0:
        raise ValueError('clock_seconds must be finite and positive')
    if type(max_plies) is not int or max_plies < 0:
        raise ValueError('max_plies must be a nonnegative integer')
    if type(claim_draw) is not bool:
        raise ValueError('claim_draw must be boolean')
    board = chess.Board(initial_fen)
    _standard_board(board)
    clock = time.perf_counter if timer is None else timer
    initial = {'white': float(clock_seconds), 'black': float(clock_seconds)}
    game = GameResult(board.fen(), white_name, black_name, initial, dict(initial), claim_draw, max_plies)

    while True:
        outcome = board.outcome(claim_draw=claim_draw)
        if outcome is not None:
            game.result = outcome.result()
            game.termination = outcome.termination.name.lower()
            game.status = 'completed'
            break
        if len(game.moves) >= max_plies:
            break
        side = 'white' if board.turn else 'black'
        legal_ids = {move.uci() for move in board.legal_moves}
        before = board.fen()
        clocks_before = dict(game.clocks)
        policy = white if board.turn else black
        response, error = None, None
        start = clock()
        try:
            # Policies cannot corrupt the authoritative board or its history.
            response = policy(board.copy(stack=True))
        except Exception as exc:  # noqa: BLE001 - A failed external policy must leave a terminal trace.
            error = {'type': type(exc).__name__, 'message': str(exc)}
        elapsed = clock()-start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Timer must produce finite, monotonic elapsed times')
        game.clocks[side] = max(0., game.clocks[side]-elapsed)
        parsed = None
        response_snapshot = None
        validation_error = None
        if error is None:
            try:
                if isinstance(response, Mapping):
                    response_snapshot = json.loads(json.dumps(dict(response), allow_nan=False))
                parsed = _response(response, legal_ids)
            except (TypeError, ValueError, OverflowError) as exc:
                validation_error = {'type': type(exc).__name__, 'message': str(exc)}
        attempt = {
            'ply': len(game.moves)+1, 'side': side, 'fen_before': before, 'fen_after': before,
            'move_uci': (response_snapshot.get('choice') if response_snapshot is not None else None),
            'san': None,
            'latency_ms': elapsed*1000, 'wall_ms': elapsed*1000,
            'clocks_before': clocks_before, 'clocks': dict(game.clocks),
            'policy': response_snapshot, 'played': False,
        }
        if error is not None or validation_error is not None:
            attempt['error'] = error or validation_error
        game.attempts.append(attempt)
        if game.clocks[side] <= 0:
            if board.has_insufficient_material(not board.turn):
                game.result = '1/2-1/2'
                game.termination = 'timeout_insufficient_material'
            else:
                game.result = '0-1' if board.turn else '1-0'
                game.termination = 'timeout'
            game.status = 'completed'
            attempt['disposition'] = 'discarded_timeout'
            break
        if error is not None or validation_error is not None:
            game.termination = 'policy_error' if error is not None else 'invalid_response'
            game.status = 'failed'
            attempt['disposition'] = game.termination
            break
        if parsed['choice'] not in legal_ids:
            game.termination = 'invalid_choice'
            game.status = 'failed'
            attempt['disposition'] = 'invalid_choice'
            break
        move = chess.Move.from_uci(parsed['choice'])
        attempt['san'] = board.san(move)
        board.push(move)
        attempt.update({'fen_after': board.fen(), 'played': True, 'disposition': 'played'})
        game.moves.append(dict(attempt))
    game.final_fen = board.fen()
    return game
