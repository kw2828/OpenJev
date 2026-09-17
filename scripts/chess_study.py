"""Frozen, configurable chess development pilot. Never trains or retries policies.

Prepare selects 24 public Lichess positions before inference, binds local model
artifacts and source files, and freezes the game schedule. Run can execute the
whole schedule or one named game in a fresh directory. Separate executions are
explicit subsets, not resumptions or a claim that the entire panel completed.

Factories receive the frozen policy specification and return a board callable.
Additional local or external policy routes can register in POLICY_FACTORIES;
their source/artifact dependencies must be included in the prepared config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.engine

from openjev.research.chess_arena import (
    _response,
    greedy_material_policy,
    play_game,
    random_policy,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'openjev-chess-development-v1'
PUZZLE_SOURCE = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'
BANDS = ('at_most_1200', '1201_to_1800', 'above_1800')
CORE_SOURCES = (
    'scripts/chess_study.py', 'src/openjev/research/chess_arena.py',
    'src/openjev/research/chess_scorer.py', 'tests/test_chess_study.py',
    'tests/test_chess_arena.py', 'tests/test_chess_scorer.py',
    'src/openjev/decisions.py', 'pyproject.toml', 'uv.lock',
)
PACKAGES = ('chess', 'python-chess', 'numpy', 'mlx', 'mlx-lm', 'torch', 'transformers', 'safetensors',
            'tokenizers', 'huggingface-hub')


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def dependencies() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def rating_band(rating: int) -> str:
    return BANDS[0 if rating <= 1200 else 1 if rating <= 1800 else 2]


def parse_puzzle(row: Mapping[str, str], source_line: int) -> dict[str, Any]:
    """Validate the complete line, then apply Lichess's opponent setup move."""
    if not row.get('PuzzleId') or None in row or any(value is None for value in row.values()):
        raise ValueError('Incomplete CSV row')
    board = chess.Board(row['FEN'])
    if not board.is_valid() or board.chess960:
        raise ValueError('Invalid standard-chess FEN')
    moves = row['Moves'].split()
    if len(moves) < 2:
        raise ValueError('Puzzle needs a setup move and solver move')
    checked = board.copy(stack=True)
    for token in moves:
        move = chess.Move.from_uci(token)
        if move not in checked.legal_moves:
            raise ValueError(f'Illegal solution move {token}')
        checked.push(move)
    board.push_uci(moves[0])
    if board.is_game_over() or not 1 <= board.legal_moves.count() <= 218:
        raise ValueError('Solver position must have a legal decision')
    rating = int(row['Rating'])
    return {
        'puzzle_id': row['PuzzleId'], 'source_line': source_line,
        'source_fen': row['FEN'], 'setup_uci': moves[0], 'solver_fen': board.fen(),
        'gold_first_uci': moves[1], 'gold_solution_uci': moves[1:],
        'rating': rating, 'rating_band': rating_band(rating),
        'source_game': row['GameUrl'], 'themes': row.get('Themes', ''),
    }


def select_puzzles(path: Path, per_band: int = 8) -> dict[str, Any]:
    """First valid rows per rating band, independent of model or engine outputs.

    A bounded decompressed archive may end mid-row. Only newline-terminated CSV
    rows are parsed; the discarded suffix and all rejected encountered rows are
    accounted for. Selection is deliberately a convenience prefix, not random.
    """
    if type(per_band) is not int or per_band <= 0:
        raise ValueError('per_band must be positive')
    raw = path.read_bytes()
    cut = raw.rfind(b'\n') + 1
    text = raw[:cut].decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))
    required = {'PuzzleId', 'FEN', 'Moves', 'Rating', 'GameUrl'}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError('Missing official Lichess CSV columns')
    selected, rejected, seen = [], [], set()
    counts = dict.fromkeys(BANDS, 0)
    for source_line, row in enumerate(reader, 2):
        try:
            puzzle = parse_puzzle(row, source_line)
            if puzzle['puzzle_id'] in seen:
                raise ValueError('Duplicate puzzle ID')
            seen.add(puzzle['puzzle_id'])
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'source_line': source_line, 'puzzle_id': row.get('PuzzleId'), 'reason': str(exc)})
            continue
        band = puzzle['rating_band']
        if counts[band] < per_band:
            selected.append(puzzle)
            counts[band] += 1
        if all(count == per_band for count in counts.values()):
            break
    if any(count != per_band for count in counts.values()):
        raise ValueError(f'Insufficient valid puzzle rows: {counts}, required {per_band} per band')
    return {
        'positions': selected, 'counts': counts, 'encountered_rejections': rejected,
        'discarded_incomplete_suffix_bytes': len(raw) - cut,
        'selection': f'first {per_band} valid puzzles per rating band in CSV source order',
        'source_url': PUZZLE_SOURCE, 'license': 'CC0',
        'scope': 'public convenience prefix; development only; possible training overlap',
        'metric': 'strict first-solver-move gold match, not full puzzle solved',
        'secondary_metric': 'gold match or an immediate legal checkmate',
    }


class StockfishPolicy:
    """Fixed-node reference; fresh hash and game identity for every decision."""

    def __init__(self, spec: Mapping[str, Any]):
        self.nodes = int(spec.get('nodes', 1000))
        self.hash_mb = int(spec.get('hash_mb', 16))
        if self.nodes <= 0 or self.hash_mb <= 0:
            raise ValueError('Stockfish nodes and hash_mb must be positive')
        self.engine = chess.engine.SimpleEngine.popen_uci(str(spec['engine_path']))
        self.engine.configure({'Threads': 1, 'Hash': self.hash_mb})
        self.metadata = {
            'kind': 'stockfish', 'engine_id': dict(self.engine.id), 'nodes': self.nodes,
            'threads': 1, 'hash_mb': self.hash_mb, 'fresh_hash_each_decision': True,
            'timing_route': 'local UCI subprocess',
        }

    def __call__(self, board: chess.Board) -> dict[str, Any]:
        self.engine.configure({'Clear Hash': None})
        result = self.engine.play(board, chess.engine.Limit(nodes=self.nodes), game=object())
        if result.move is None:
            raise ValueError('Stockfish returned no move')
        return {'choice': result.move.uci(), 'policy_kind': 'stockfish_fixed_nodes', 'nodes': self.nodes}

    def close(self) -> None:
        self.engine.quit()


def qwen_factory(spec: Mapping[str, Any]) -> Any:
    from openjev.research.chess_scorer import ChessMLXPolicy
    # The scorer itself owns the pinned model identity and loading route. The
    # specification's model_path is bound as an artifact, not an override.
    return ChessMLXPolicy()


def lfm_factory(spec: Mapping[str, Any]) -> Any:
    from openjev.research.chess_lfm import ChessLFMPolicy
    return ChessLFMPolicy(spec['model_path'], device=spec.get('device', 'cpu'), threads=spec.get('threads', 2))


def chessfly_factory(spec: Mapping[str, Any]) -> Any:
    from openjev.research.chess_fly import ChessFlyPolicy
    return ChessFlyPolicy(spec['asset_dir'], threads=spec.get('threads', 2))


def astra_factory(spec: Mapping[str, Any]) -> Any:
    from openjev.research.chess_external import CodexChessPolicy
    return CodexChessPolicy(spec['bridge_dir'], timeout_seconds=spec.get('timeout_seconds', 120))


def validate_student_spec(spec: Mapping[str, Any]) -> None:
    """Require an explicit frozen checkpoint identity, never a best-seed route."""
    if not spec.get('checkpoint_path'):
        raise ValueError('Chess student requires checkpoint_path')
    digest = spec.get('expected_training_plan_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Chess student requires a lowercase SHA256 expected_training_plan_sha256')
    if spec.get('mode') not in ('gru', 'circuit', 'rewired'):
        raise ValueError('Chess student requires an explicit supported mode')
    if type(spec.get('seed')) is not int or spec['seed'] not in (17, 29, 43):
        raise ValueError('Chess student requires an explicit frozen fit seed: 17, 29 or 43')
    if spec.get('device', 'cpu') != 'cpu' or type(spec.get('threads', 2)) is not int or spec.get('threads', 2) != 2:
        raise ValueError('Chess student benchmark uses CPU with exactly two threads')


class ChessStudentPolicy:
    """Load exactly one declared final student checkpoint, without selection."""

    def __init__(self, spec: Mapping[str, Any]):
        import torch

        from openjev.research.chess_student import DEPTH, MOVE_VOCAB_SHA256, ChessStudent

        validate_student_spec(spec)
        checkpoint = Path(spec['checkpoint_path']).expanduser().resolve()
        torch.set_num_threads(2)
        self.model = ChessStudent.load(checkpoint, expected_plan_sha256=spec['expected_training_plan_sha256'])
        if self.model.mode != spec['mode'] or self.model.seed != spec['seed']:
            raise ValueError('Loaded chess student mode or seed differs from the declared checkpoint identity')
        self.metadata = {
            'kind': 'chess_student', 'model': f'openjev-chess-student/{self.model.mode}',
            'revision': spec['expected_training_plan_sha256'],
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'training_plan_sha256': spec['expected_training_plan_sha256'],
            'mode': self.model.mode, 'seed': self.model.seed, 'depth': DEPTH,
            'device': 'cpu', 'threads': 2, 'vocabulary_sha256': MOVE_VOCAB_SHA256,
            'parameter_count': self.model.parameter_count(),
            'checkpoint_selection': 'Exact named final checkpoint; no seed or metric selection',
            'timing_route': 'local CPU numeric FEN encoder and depth-four policy',
            'scope': 'Stockfish-distilled compact model; circuits are synthetic, not imported connectomes',
        }

    def __call__(self, board: chess.Board) -> dict[str, Any]:
        return self.model.choose(board)


def chess_student_factory(spec: Mapping[str, Any]) -> Any:
    return ChessStudentPolicy(spec)


POLICY_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    'random': lambda spec: random_policy(int(spec.get('seed', 20260917))),
    'greedy': lambda spec: greedy_material_policy(),
    'stockfish': StockfishPolicy,
    'qwen': qwen_factory,
    'lfm': lfm_factory,
    'chessfly': chessfly_factory,
    'astra_codex': astra_factory,
    'chess_student': chess_student_factory,
}


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate only declared routes. Unknown kinds fail before any inference."""
    config = json.loads(json.dumps(config, allow_nan=False))
    policies = config.get('policies')
    if not isinstance(policies, dict) or not policies:
        raise ValueError('Declare named policies')
    for name, spec in policies.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError('Policy names and specifications must be nonempty mappings')
        if spec.get('kind') not in POLICY_FACTORIES:
            raise ValueError(f'Unregistered policy kind: {spec.get("kind")}')
        for key in ('model_path', 'engine_path', 'asset_dir', 'bridge_dir', 'checkpoint_path'):
            if key in spec:
                spec[key] = str(Path(spec[key]).expanduser().resolve())
        spec['artifacts'] = [str(Path(path).expanduser().resolve()) for path in spec.get('artifacts', [])]
        spec.setdefault('warmup_calls', 2 if spec['kind'] in ('qwen', 'lfm', 'chessfly', 'chess_student') else 0)
        if type(spec['warmup_calls']) is not int or not 0 <= spec['warmup_calls'] <= 2:
            raise ValueError('warmup_calls must be 0, 1 or 2, declared before inference')
        if spec['kind'] in ('qwen', 'lfm') and (not spec.get('model_path') or not spec.get('revision')):
            raise ValueError('Learned model requires local model_path and declared revision')
        if spec['kind'] == 'chessfly' and not spec.get('asset_dir'):
            raise ValueError('ChessFly requires local asset_dir')
        if spec['kind'] == 'astra_codex' and not spec.get('bridge_dir'):
            raise ValueError('Codex Astra requires a fresh bridge_dir')
        if spec['kind'] == 'stockfish' and not spec.get('engine_path'):
            raise ValueError('Stockfish requires engine_path')
        if spec['kind'] == 'chess_student':
            validate_student_spec(spec)
            spec.setdefault('device', 'cpu')
            spec.setdefault('threads', 2)
    config.setdefault('clock_seconds', 300.)
    config.setdefault('max_plies', 120)
    config.setdefault('claim_draw', False)
    if (type(config['clock_seconds']) not in (int, float) or not math.isfinite(config['clock_seconds'])
            or config['clock_seconds'] <= 0):
        raise ValueError('Positive finite clocks required')
    if type(config['max_plies']) is not int or config['max_plies'] < 1:
        raise ValueError('Positive max_plies required')
    if type(config['claim_draw']) is not bool:
        raise ValueError('claim_draw must be boolean')
    config.setdefault('games', [])
    seen = set()
    for game in config['games']:
        if not isinstance(game, dict) or set(game) != {'id', 'white', 'black'}:
            raise ValueError('Each game specifies exactly id, white and black')
        game_id = game['id']
        if (not isinstance(game_id, str) or not game_id or not all(
                char.isascii() and (char.isalnum() or char in '_-') for char in game_id)):
            raise ValueError('Game IDs must be safe filename components')
        if game_id in seen or game['white'] not in policies or game['black'] not in policies:
            raise ValueError('Duplicate game ID or unknown player')
        seen.add(game_id)
    config.setdefault('puzzle_policies', list(policies))
    if (len(set(config['puzzle_policies'])) != len(config['puzzle_policies'])
            or any(name not in policies for name in config['puzzle_policies'])):
        raise ValueError('Puzzle policy names must be distinct declared policies')
    if not config['games'] and not config['puzzle_policies']:
        raise ValueError('Empty execution schedule')
    config.setdefault('extra_sources', [])
    return config


def file_bindings(paths: list[Path]) -> dict[str, str]:
    files = {}
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            contents = sorted(item for item in path.rglob('*') if item.is_file() and '.git' not in item.parts)
            if not contents:
                raise ValueError(f'Empty artifact directory: {path}')
        elif path.is_file():
            contents = [path]
        else:
            raise FileNotFoundError(path)
        for item in contents:
            files[str(item)] = sha256(item)
    return files


def prepare(config: dict[str, Any], puzzles: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    config = normalize_config(config)
    source_paths = [ROOT / path for path in CORE_SOURCES]
    for spec in config['policies'].values():
        if spec['kind'] == 'astra_codex':
            source_paths += [ROOT / 'src/openjev/research/chess_external.py',
                             ROOT / 'tests/test_chess_external.py', ROOT / 'scripts/chess_agent_bridge.py']
        if spec['kind'] in ('lfm', 'chessfly'):
            module = 'chess_lfm' if spec['kind'] == 'lfm' else 'chess_fly'
            source_paths += [ROOT / f'src/openjev/research/{module}.py', ROOT / f'tests/test_{module}.py']
        if spec['kind'] == 'chess_student':
            source_paths += [ROOT / 'src/openjev/research/chess_student.py',
                             ROOT / 'scripts/train_chess_student.py', ROOT / 'tests/test_chess_student.py']
    source_paths += [ROOT / path for path in config['extra_sources']]
    artifact_paths = [puzzles]
    for spec in config['policies'].values():
        artifact_paths += [Path(spec[key]) for key in
                           ('model_path', 'engine_path', 'asset_dir', 'checkpoint_path') if key in spec]
        artifact_paths += [Path(path) for path in spec['artifacts']]
    plan = {
        'format': FORMAT, 'prepared_utc': datetime.now(UTC).isoformat(), 'config': config,
        'puzzle_csv': str(puzzles.resolve()), 'puzzles': select_puzzles(puzzles),
        'source_hashes': file_bindings(source_paths), 'artifact_hashes': file_bindings(artifact_paths),
        'artifact_roots': [str(path.resolve()) for path in artifact_paths],
        'dependencies': dependencies(),
        'claim_boundary': (
            'Development pilot, not Elo or a reproduction of another model tournament. '
            'Legal moves are supplied by python-chess. Fixed nodes are not matched compute. '
            'Public puzzles may overlap model training. Candidate probabilities are not win probabilities. '
            'Clock and ply-cap outcomes remain separate; capped games are unfinished.'
        ),
    }
    plan['plan_id'] = canonical_hash(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, plan)
    return plan


def verify_plan(plan: dict[str, Any]) -> None:
    if plan.get('format') != FORMAT:
        raise ValueError('Unknown plan format')
    contents = {key: value for key, value in plan.items() if key != 'plan_id'}
    if canonical_hash(contents) != plan.get('plan_id'):
        raise ValueError('Plan identity mismatch')
    if normalize_config(plan['config']) != plan['config']:
        raise ValueError('Unnormalized plan')
    if dependencies() != plan['dependencies']:
        raise ValueError('Execution dependencies differ from prepared environment')
    for group in ('source_hashes', 'artifact_hashes'):
        for path, expected in plan[group].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f'Changed {group}: {path}')
    actual_artifacts = file_bindings([Path(path) for path in plan['artifact_roots']])
    if actual_artifacts != plan['artifact_hashes']:
        raise ValueError('Artifact directory membership changed')


def policy_metadata(policy: Any) -> Any:
    metadata = getattr(policy, 'metadata', {})
    return metadata() if callable(metadata) else metadata


def timed_choice(policy: Any, board: chess.Board) -> dict[str, Any]:
    started = time.perf_counter()
    response = error = raw_response = None
    try:
        response = policy(board.copy(stack=True))
    except Exception as exc:  # noqa: BLE001 - Preserve failed calls without retry.
        error = {'type': type(exc).__name__, 'message': str(exc)}
    elapsed = time.perf_counter() - started
    if error is None:
        try:
            raw_response = json.loads(json.dumps(response, allow_nan=False))
            response = _response(response, {move.uci() for move in board.legal_moves})
            if response['choice'] not in {move.uci() for move in board.legal_moves}:
                raise ValueError('Policy chose an illegal move')
        except (ValueError, TypeError, OverflowError) as exc:
            response = None
            error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'response': response, 'raw_response': raw_response, 'error': error, 'wall_ms': elapsed * 1000}


def evaluate_puzzle(policy: Any, puzzle: Mapping[str, Any]) -> dict[str, Any]:
    board = chess.Board(puzzle['source_fen'])
    board.push_uci(puzzle['setup_uci'])
    if board.fen() != puzzle['solver_fen']:
        raise ValueError('Prepared solver FEN mismatch')
    record = timed_choice(policy, board)
    choice = record['response']['choice'] if record['response'] else None
    gold_match = choice == puzzle['gold_first_uci']
    mate = False
    if choice is not None:
        board.push_uci(choice)
        mate = board.is_checkmate()
    return {
        'puzzle_id': puzzle['puzzle_id'], 'rating_band': puzzle['rating_band'],
        'solver_fen': puzzle['solver_fen'], 'gold_first_uci': puzzle['gold_first_uci'],
        'gold_match': gold_match, 'immediate_checkmate': mate,
        'gold_or_immediate_mate': gold_match or mate, **record,
    }


def latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'calls': 0, 'mean_ms': None, 'median_ms': None, 'p95_ms': None, 'total_ms': 0.}
    ordered = sorted(values)
    return {
        'calls': len(values), 'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
        'p95_ms': ordered[math.ceil(.95 * len(values)) - 1], 'total_ms': sum(values),
    }


def summarize(puzzles: list[dict[str, Any]], games: list[dict[str, Any]]) -> dict[str, Any]:
    puzzle_summary = {}
    for name in dict.fromkeys(row['policy_name'] for row in puzzles):
        rows = [row for row in puzzles if row['policy_name'] == name]
        puzzle_summary[name] = {
            'positions': len(rows), 'strict_gold_matches': sum(row['gold_match'] for row in rows),
            'gold_or_immediate_mate': sum(row['gold_or_immediate_mate'] for row in rows),
            'errors': sum(row['error'] is not None for row in rows),
            'accuracy': sum(row['gold_match'] for row in rows) / len(rows),
            'latency_all_attempts': latency_summary([row['wall_ms'] for row in rows]),
            'by_rating_band': {
                band: {'positions': sum(row['rating_band'] == band for row in rows),
                       'strict_gold_matches': sum(row['gold_match'] and row['rating_band'] == band for row in rows)}
                for band in BANDS
            },
        }
    game_rows, calls = [], {}
    for game in games:
        game_rows.append({key: game[key] for key in (
            'game_id', 'white', 'black', 'result', 'status', 'termination', 'clocks',
        )} | {'played_plies': len(game['moves'])})
        for attempt in game['attempts']:
            name = game[attempt['side']]
            calls.setdefault(name, []).append(attempt['wall_ms'])
    return {
        'puzzles': puzzle_summary, 'games': game_rows,
        'game_policy_latency_all_attempts': {name: latency_summary(values) for name, values in calls.items()},
        'scored_game_count': sum(game['status'] == 'completed' for game in games),
        'unfinished_game_count': sum(game['status'] == 'unfinished' for game in games),
        'failed_game_count': sum(game['status'] == 'failed' for game in games),
        'latency_scope': 'policy-call wall time; initial model load and declared warmups excluded',
    }


def run(plan_path: Path, out: Path, *, game_id: str | None = None, puzzles_only: bool = False) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    if game_id is not None and puzzles_only:
        raise ValueError('Select either a game or puzzles-only')
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    config = plan['config']
    games = config['games'] if not puzzles_only else []
    if game_id is not None:
        games = [game for game in games if game['id'] == game_id]
        if not games:
            raise ValueError(f'Unknown game ID: {game_id}')
    puzzle_names = config['puzzle_policies'] if game_id is None else []
    names = list(dict.fromkeys(puzzle_names + [game[side] for game in games for side in ('white', 'black')]))
    out.mkdir(parents=True)
    write_json(out / 'plan.json', plan)
    selection = {'game_ids': [game['id'] for game in games], 'puzzle_policies': puzzle_names}
    write_json(out / 'started.json', {
        'plan_id': plan['plan_id'], 'plan_file_sha256': sha256(plan_path),
        'started_utc': datetime.now(UTC).isoformat(), 'selection': selection,
    })
    policies, loads, warmups, puzzle_records, game_records = {}, {}, [], [], []
    started = time.perf_counter()
    try:
        for name in names:
            spec = config['policies'][name]
            load_start = time.perf_counter()
            policies[name] = POLICY_FACTORIES[spec['kind']](spec)
            metadata = policy_metadata(policies[name])
            if spec.get('revision') and metadata.get('revision') != spec['revision']:
                raise ValueError(f'Loaded revision does not match prepared policy {name}')
            if spec.get('model_id') and metadata.get('model') != spec['model_id']:
                raise ValueError(f'Loaded model does not match prepared policy {name}')
            loads[name] = {'wall_seconds': time.perf_counter() - load_start,
                           'metadata': metadata, 'declared_spec': spec}
            for number in range(spec['warmup_calls']):
                record = timed_choice(policies[name], chess.Board())
                warmups.append({'policy_name': name, 'warmup_index': number, **record})
                if record['error'] is not None:
                    raise ValueError(f'Declared warmup failed for {name}: {record["error"]}')
        write_json(out / 'loads.json', loads)
        write_json(out / 'warmups.json', warmups)
        with (out / 'puzzles.jsonl').open('x') as handle:
            for puzzle in plan['puzzles']['positions']:
                for name in puzzle_names:
                    record = {'policy_name': name, **evaluate_puzzle(policies[name], puzzle)}
                    puzzle_records.append(record)
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                    handle.flush()
        for game_spec in games:
            result = play_game(
                policies[game_spec['white']], policies[game_spec['black']],
                white_name=game_spec['white'], black_name=game_spec['black'],
                initial_fen=chess.STARTING_FEN, clock_seconds=config['clock_seconds'],
                max_plies=config['max_plies'], claim_draw=config['claim_draw'],
            )
            record = {'game_id': game_spec['id'], **result.to_dict()}
            game_records.append(record)
            write_json(out / f'game-{game_spec["id"]}.json', record)
            with (out / f'game-{game_spec["id"]}.pgn').open('x') as handle:
                handle.write(result.to_pgn() + '\n')
        verify_plan(plan)
        summary = {
            'format': FORMAT, 'plan_id': plan['plan_id'], 'selection': selection,
            'entire_prepared_panel': game_id is None and not puzzles_only,
            'elapsed_seconds': time.perf_counter() - started,
            'claim_boundary': plan['claim_boundary'], **summarize(puzzle_records, game_records),
        }
        write_json(out / 'summary.json', summary)
        write_json(out / 'completed.json', {
            'plan_id': plan['plan_id'], 'status': 'selected_schedule_executed',
            'selection': selection, 'completed_utc': datetime.now(UTC).isoformat(),
            'puzzle_attempts': len(puzzle_records), 'games_executed': len(game_records),
            'outputs': {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()},
        })
        return summary
    except BaseException as exc:
        write_json(out / 'failed.json', {
            'plan_id': plan['plan_id'], 'type': type(exc).__name__, 'message': str(exc),
            'puzzle_attempts': len(puzzle_records), 'games_executed': len(game_records),
            'loads': loads, 'warmups': warmups,
        })
        raise
    finally:
        for policy in policies.values():
            close = getattr(policy, 'close', None)
            if callable(close):
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--puzzles', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True, help='New plan JSON path')
    execute = commands.add_parser('run')
    execute.add_argument('--plan', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True, help='New execution directory')
    select = execute.add_mutually_exclusive_group()
    select.add_argument('--game-id')
    select.add_argument('--puzzles-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(json.loads(args.config.read_text()), args.puzzles, args.out)
        print(json.dumps({'plan_id': result['plan_id'], 'positions': len(result['puzzles']['positions'])}))
    else:
        print(json.dumps(run(args.plan, args.out, game_id=args.game_id, puzzles_only=args.puzzles_only)))


if __name__ == '__main__':
    main()
