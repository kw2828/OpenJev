"""Bounded, auditable chess positions and exact action-conditioned transitions.

Every visited nonterminal position makes one teacher call, including excluded
and duplicate states. Auxiliary actions use a separate deterministic RNG and
python-chess transitions, never additional engine calls or teacher move hints.
"""

import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import chess
import chess.engine

DEFAULT_CONFIG = {
    'teacher_nodes': 2000, 'value_cp_scale': 600., 'mate_cp': 10000,
    'teacher_threads': 1, 'teacher_hash_mb': 16, 'aux_actions': 4,
    'splits': [
        {'name': 'train', 'examples': 32768, 'game_cap': 1200, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 92_000_000},
        {'name': 'dev', 'examples': 4096, 'game_cap': 200, 'max_plies': 64,
         'random_move_probability': .5, 'seed_base': 93_000_000},
        {'name': 'shift', 'examples': 4096, 'game_cap': 200, 'max_plies': 96,
         'random_move_probability': .1, 'seed_base': 94_000_000},
    ],
}


def state_key(board):
    """Keep placement, turn, castling and literal EP; omit both move counters."""
    return ' '.join(board.fen(en_passant='fen').split()[:4])


def symmetry_key(board):
    """Identify the side-canonical equivalence class used by the spatial model."""
    return min(state_key(board), state_key(board.mirror()))


def _exclusion_keys(exclusions):
    reserved = set()
    canonical = set()
    for key in exclusions:
        if not isinstance(key, str) or len(key.split()) != 4:
            raise ValueError('Excluded states must be four-field FEN state keys')
        board = chess.Board(key+' 0 1')
        if not board.is_valid() or key != state_key(board):
            raise ValueError('Excluded state is not a valid normalized standard chess state')
        reserved.update((key, state_key(board.mirror())))
        canonical.add(symmetry_key(board))
    return canonical, reserved


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _append(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n')
    stream.flush()


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _positive_int(value):
    return type(value) is int and value > 0


def _validate_config(config):
    if not _positive_int(config['teacher_nodes']) or not _positive_int(config['mate_cp']):
        raise ValueError('Teacher nodes and mate centipawns must be positive integers')
    if not math.isfinite(config['value_cp_scale']) or config['value_cp_scale'] <= 0:
        raise ValueError('Value centipawn scale must be positive and finite')
    if config.get('aux_actions', 4) != 4:
        raise ValueError('This protocol requires four distinct auxiliary moves when available')
    if config.get('teacher_threads', 1) != 1 or config.get('teacher_hash_mb', 16) != 16:
        raise ValueError('This protocol requires one teacher thread and 16 MB hash')
    if not config['splits']:
        raise ValueError('At least one data split is required')
    names = set()
    for split in config['splits']:
        name = split['name']
        if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]*', name) or name in names:
            raise ValueError('Split names must be unique safe lowercase names')
        names.add(name)
        if name in {'analyses', 'games'}:
            raise ValueError('Split name collides with an evidence file')
        if any(not _positive_int(split[key]) for key in ('examples', 'game_cap', 'max_plies')):
            raise ValueError('Split counts and caps must be positive integers')
        if type(split['seed_base']) is not int:
            raise ValueError('Split seed base must be an integer')
        if not 0 <= split['random_move_probability'] <= 1:
            raise ValueError('Random rollout probability must lie in [0,1]')


def _validate_label(label, board, config):
    if not isinstance(label, dict):
        raise TypeError('Teacher label must be a dictionary')
    target = chess.Move.from_uci(label['target_uci'])
    if target not in board.legal_moves:
        raise ValueError('Teacher returned an illegal move')
    if type(label['score_cp']) is not int:
        raise ValueError('Teacher score must be integer centipawns')
    if label['mate'] is not None and type(label['mate']) is not int:
        raise ValueError('Mate distance must be an integer or null')
    value = label['target_value']
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Teacher value must be finite and in [-1,1]')
    if not math.isclose(value, math.tanh(label['score_cp']/config['value_cp_scale']), abs_tol=1e-12):
        raise ValueError('Teacher value disagrees with its centipawn target')
    if label['requested_nodes'] != config['teacher_nodes']:
        raise ValueError('Teacher requested-node budget differs from the frozen configuration')
    if type(label['reported_nodes']) is not int or label['reported_nodes'] < 0:
        raise ValueError('Teacher reported nodes must be a nonnegative integer')
    if not math.isfinite(label['wall_seconds']) or label['wall_seconds'] < 0:
        raise ValueError('Teacher wall time must be nonnegative and finite')
    return {key: label[key] for key in ('target_uci', 'target_value', 'score_cp', 'mate',
                                      'requested_nodes', 'reported_nodes', 'wall_seconds')}


def stockfish_label(engine, board, config):
    """Label a history-free FEN; caller configures Threads=1 and Hash=16 once.

    Clearing the transposition table is included in wall time. The node limit
    is a requested limit; the engine's actual reported node count is retained.
    """
    started = time.perf_counter()
    engine.configure({'Clear Hash': None})
    position = chess.Board(board.fen(en_passant='fen'))
    info = engine.analyse(position, chess.engine.Limit(nodes=config['teacher_nodes']),
                          info=chess.engine.INFO_SCORE | chess.engine.INFO_PV | chess.engine.INFO_BASIC)
    elapsed = time.perf_counter()-started
    move = info['pv'][0]
    score = info['score'].pov(position.turn)
    cp = score.score(mate_score=config['mate_cp'])
    if cp is None:
        raise ValueError('Teacher returned no numeric score')
    label = {'target_uci': move.uci(), 'score_cp': int(cp), 'mate': score.mate(),
             'target_value': math.tanh(cp/config['value_cp_scale']),
             'reported_nodes': int(info['nodes']), 'requested_nodes': config['teacher_nodes'],
             'wall_seconds': elapsed}
    return _validate_label(label, position, config)


def auxiliary_transitions(board, game_seed, ply):
    """Sample without replacement, independently of the behavior-policy RNG."""
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    material = f'aux-v1|{game_seed}|{ply}|{board.fen(en_passant="fen")}'.encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    selected = rng.sample(legal, 4) if len(legal) >= 4 else legal
    transitions = []
    for move in selected:
        successor = board.copy(stack=False)
        successor.push(move)
        transitions.append({'uci': move.uci(), 'next_fen': successor.fen(en_passant='fen')})
    return transitions


def _call_teacher(labeler, board, config, analyses, counters, metadata):
    """Record the attempt before propagating errors; never retry a failed call."""
    counters['teacher_calls'] += 1
    counters['requested_nodes'] += config['teacher_nodes']
    started = time.perf_counter()
    raw = None
    try:
        # A caller-owned labeler cannot change the rollout position or history.
        raw = labeler(chess.Board(board.fen(en_passant='fen')))
        label = _validate_label(raw, board, config)
    except Exception as exc:
        elapsed = time.perf_counter()-started
        counters['teacher_call_wall_seconds'] += elapsed
        record = {**metadata, 'status': 'failed', 'requested_nodes': config['teacher_nodes'],
                  'reported_nodes': None, 'wall_seconds': None, 'call_wall_seconds': elapsed,
                  'error_type': type(exc).__name__, 'error': str(exc)}
        if isinstance(raw, dict):
            nodes, wall = raw.get('reported_nodes'), raw.get('wall_seconds')
            if type(nodes) is int and nodes >= 0:
                record['reported_nodes'] = nodes
                counters['reported_nodes'] += nodes
                counters['known_node_calls'] += 1
            if type(wall) in (int, float) and math.isfinite(wall) and wall >= 0:
                record['wall_seconds'] = wall
                counters['teacher_wall_seconds'] += wall
        _append(analyses, record)
        raise
    elapsed = time.perf_counter()-started
    counters['successful_teacher_calls'] += 1
    counters['known_node_calls'] += 1
    counters['reported_nodes'] += label['reported_nodes']
    counters['teacher_wall_seconds'] += label['wall_seconds']
    counters['teacher_call_wall_seconds'] += elapsed
    _append(analyses, {**metadata, 'status': 'completed', **label, 'call_wall_seconds': elapsed})
    return label


def generate_data(out, labeler, config, excluded_states):
    """Generate one fixed-budget attempt and return its completed receipt.

    A failure writes failed.json and retains all partial evidence, then raises.
    The output directory must not exist. Exclusions are durable state keys from
    earlier studies; all splits share a second, initially empty deduplication set.
    """
    _validate_config(config)
    exclusions = set(excluded_states)
    excluded_classes, reserved = _exclusion_keys(exclusions)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    _write_new(out/'started.json', {'started_unix': started, 'config': config})
    _write_new(out/'excluded-states.json', sorted(exclusions))
    seen = set()
    counts = {split['name']: 0 for split in config['splits']}
    counters = dict.fromkeys(('teacher_calls', 'successful_teacher_calls', 'requested_nodes',
                              'reported_nodes', 'known_node_calls', 'generated_games'), 0)
    counters.update(teacher_wall_seconds=0., teacher_call_wall_seconds=0.)

    def receipt(status):
        return {'status': status, 'started_unix': started, 'completed_unix': time.time(),
                'counts': counts.copy(), 'unique_states': len(seen), 'excluded_states': len(exclusions),
                'excluded_symmetry_classes': len(excluded_classes), 'reserved_state_keys': len(reserved),
                **counters, 'unknown_node_calls': counters['teacher_calls']-counters['known_node_calls'],
                'rollout_only_teacher_calls': counters['successful_teacher_calls']-len(seen),
                'teacher_cost_scope': 'One call per visited nonterminal state, including duplicates/exclusions',
                'auxiliary_cost_scope': 'Four uniform distinct legal transitions from python-chess; no engine',
                'files': {path.name: sha(path) for path in sorted(out.iterdir())
                          if path.suffix in {'.json', '.jsonl'} and path.name not in {'completed.json',
                                                                                  'failed.json'}}}

    try:
        with (out/'games.jsonl').open('x') as games, (out/'analyses.jsonl').open('x') as analyses:
            game_offset = 0
            for split in config['splits']:
                name = split['name']
                with (out/f'{name}.jsonl').open('x') as records:
                    for local_game in range(split['game_cap']):
                        game_id, seed = game_offset+local_game, split['seed_base']+local_game
                        rng, board = random.Random(seed), chess.Board()
                        actions, sampled = [], 0
                        counters['generated_games'] += 1
                        stop = 'ply_cap'
                        try:
                            for ply in range(split['max_plies']):
                                if board.outcome(claim_draw=False) is not None:
                                    stop = 'terminal'
                                    break
                                fen, key = board.fen(en_passant='fen'), state_key(board)
                                canonical = symmetry_key(board)
                                accepted = canonical not in seen and canonical not in excluded_classes
                                label = _call_teacher(labeler, board, config, analyses, counters,
                                                      {'split': name, 'game_id': game_id, 'ply': ply,
                                                       'fen': fen, 'state_key': key,
                                                       'labelled_position': accepted,
                                                       'excluded_position': canonical in excluded_classes})
                                random_action = rng.random() < split['random_move_probability']
                                legal = sorted(board.legal_moves, key=lambda move: move.uci())
                                action = (rng.choice(legal) if random_action else
                                          chess.Move.from_uci(label['target_uci']))
                                successor = board.copy(stack=False)
                                successor.push(action)
                                if accepted:
                                    row = {'id': f'{name}-{counts[name]:05d}', 'split': name,
                                           'game_id': game_id, 'game_seed': seed, 'ply': ply, 'fen': fen,
                                           'state_key': key, 'target_uci': label['target_uci'],
                                           'target_value': label['target_value'], 'score_cp': label['score_cp'],
                                           'mate': label['mate'], 'behavior_uci': action.uci(),
                                           'next_fen': successor.fen(en_passant='fen'),
                                           'aux_transitions': auxiliary_transitions(board, seed, ply)}
                                    _append(records, row)
                                    seen.add(canonical)
                                    reserved.update((key, state_key(board.mirror())))
                                    sampled += 1
                                    counts[name] += 1
                                actions.append({'uci': action.uci(),
                                                'source': 'random' if random_action else 'teacher'})
                                board.push(action)
                                if counts[name] == split['examples']:
                                    stop = 'quota'
                                    break
                        except Exception:
                            stop = 'failed'
                            raise
                        finally:
                            _append(games, {'split': name, 'game_id': game_id, 'seed': seed,
                                            'sampled_positions': sampled, 'actions': actions,
                                            'final_fen': board.fen(en_passant='fen'), 'stop': stop})
                        if counts[name] == split['examples']:
                            break
                if counts[name] != split['examples']:
                    raise ValueError(f'{name} generated {counts[name]} unique positions before fixed game cap')
                game_offset += split['game_cap']
        result = receipt('completed')
        _write_new(out/'completed.json', result)
        return result
    except Exception as exc:
        _write_new(out/'failed.json', {**receipt('failed'), 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def validate_data(out, config, excluded_states):
    """Verify successful evidence, exact transitions, quotas and call accounting."""
    _validate_config(config)
    out = Path(out)
    result = json.loads((out/'completed.json').read_text())
    if result['status'] != 'completed' or (out/'failed.json').exists():
        raise ValueError('Data generation did not complete successfully')
    required = {'started.json', 'excluded-states.json', 'games.jsonl', 'analyses.jsonl'}
    required.update(f'{split["name"]}.jsonl' for split in config['splits'])
    if set(result['files']) != required:
        raise ValueError('Receipt does not bind the complete data evidence set')
    for name, digest in result['files'].items():
        if sha(out/name) != digest:
            raise ValueError('Data file hash mismatch')
    if json.loads((out/'started.json').read_text())['config'] != config:
        raise ValueError('Generation configuration differs from the requested configuration')
    excluded = set(excluded_states)
    excluded_classes, reserved = _exclusion_keys(excluded)
    if json.loads((out/'excluded-states.json').read_text()) != sorted(excluded):
        raise ValueError('Excluded-state manifest differs from the requested exclusions')
    seen, all_games, rows_by_split, positions, rows_by_position = set(), {}, {}, {}, {}
    split_configs, game_offset = {}, 0
    for split in config['splits']:
        split_configs[split['name']] = (split, game_offset)
        game_offset += split['game_cap']
    for game in _jsonl(out/'games.jsonl'):
        if game['game_id'] in all_games or game['stop'] == 'failed':
            raise ValueError('Duplicated or failed game')
        split, offset = split_configs[game['split']]
        local_game = game['game_id']-offset
        if (not 0 <= local_game < split['game_cap'] or
                game['seed'] != split['seed_base']+local_game or len(game['actions']) > split['max_plies']):
            raise ValueError('Game exceeds its split budget or uses an incorrect ID/seed')
        all_games[game['game_id']] = game
        board = chess.Board()
        for ply, action in enumerate(game['actions']):
            if board.outcome(claim_draw=False) is not None:
                raise ValueError('Game continued past a terminal position')
            positions[game['game_id'], ply] = board.fen(en_passant='fen')
            move = chess.Move.from_uci(action['uci'])
            if move not in board.legal_moves or action['source'] not in {'random', 'teacher'}:
                raise ValueError('Illegal game action')
            board.push(move)
        if board.fen(en_passant='fen') != game['final_fen']:
            raise ValueError('Game final state disagrees with its action trace')
    for split in config['splits']:
        name = split['name']
        rows = _jsonl(out/f'{name}.jsonl')
        if len(rows) != split['examples'] or result['counts'][name] != len(rows):
            raise ValueError('Incorrect frozen data count')
        for index, row in enumerate(rows):
            board = chess.Board(row['fen'])
            key = state_key(board)
            canonical = symmetry_key(board)
            if (not board.is_valid() or key != row['state_key'] or canonical in seen or
                    canonical in excluded_classes):
                raise ValueError('Invalid, duplicated or excluded state')
            seen.add(canonical)
            reserved.update((key, state_key(board.mirror())))
            if row['split'] != name or row['id'] != f'{name}-{index:05d}':
                raise ValueError('Incorrect row membership or ordering')
            game = all_games[row['game_id']]
            if game['split'] != name or row['game_seed'] != game['seed']:
                raise ValueError('Game IDs overlap across data splits or have an incorrect seed')
            position_id = row['game_id'], row['ply']
            if positions.get(position_id) != row['fen'] or position_id in rows_by_position:
                raise ValueError('Row position disagrees with its game action trace')
            rows_by_position[position_id] = row
            if game['actions'][row['ply']]['uci'] != row['behavior_uci']:
                raise ValueError('Row behavior move disagrees with its game action trace')
            if chess.Move.from_uci(row['target_uci']) not in board.legal_moves:
                raise ValueError('Illegal teacher move')
            if not math.isclose(row['target_value'], math.tanh(row['score_cp']/config['value_cp_scale']),
                                abs_tol=1e-12):
                raise ValueError('Inconsistent value target')
            transition = board.copy(stack=False)
            transition.push_uci(row['behavior_uci'])
            if transition.fen(en_passant='fen') != row['next_fen']:
                raise ValueError('Incorrect behavior successor')
            if row['aux_transitions'] != auxiliary_transitions(board, row['game_seed'], row['ply']):
                raise ValueError('Incorrect independently sampled auxiliary transitions')
        rows_by_split[name] = rows
    analyses = _jsonl(out/'analyses.jsonl')
    if any(call['status'] != 'completed' for call in analyses):
        raise ValueError('Completed generation contains a failed teacher call')
    if len(analyses) != sum(len(game['actions']) for game in all_games.values()):
        raise ValueError('Teacher calls do not account for every visited nonterminal position')
    visited, game_rngs, sampled_by_game = set(), {}, dict.fromkeys(all_games, 0)
    for call in analyses:
        position_id = call['game_id'], call['ply']
        if position_id in visited or positions.get(position_id) != call['fen']:
            raise ValueError('Teacher call does not match a unique visited game position')
        visited.add(position_id)
        game = all_games[call['game_id']]
        board = chess.Board(call['fen'])
        if call['split'] != game['split'] or call['state_key'] != state_key(board):
            raise ValueError('Teacher call membership or state key is incorrect')
        if call['excluded_position'] != (symmetry_key(board) in excluded_classes):
            raise ValueError('Teacher call exclusion flag is incorrect')
        if call['labelled_position'] != (position_id in rows_by_position):
            raise ValueError('Teacher call acceptance flag disagrees with stored rows')
        _validate_label(call, board, config)
        if not math.isfinite(call['call_wall_seconds']) or call['call_wall_seconds'] < 0:
            raise ValueError('Invalid recorded teacher-call wall time')
        rng = game_rngs.setdefault(call['game_id'], random.Random(game['seed']))
        split = split_configs[game['split']][0]
        random_action = rng.random() < split['random_move_probability']
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        expected_action = (rng.choice(legal).uci() if random_action else call['target_uci'])
        if game['actions'][call['ply']] != {'uci': expected_action,
                                           'source': 'random' if random_action else 'teacher'}:
            raise ValueError('Behavior action disagrees with the fixed rollout RNG and teacher')
        if position_id in rows_by_position:
            row = rows_by_position[position_id]
            if any(row[key] != call[key] for key in ('target_uci', 'target_value', 'score_cp', 'mate')):
                raise ValueError('Saved row label differs from its original teacher call')
            sampled_by_game[call['game_id']] += 1
    if any(game['sampled_positions'] != sampled_by_game[game_id] for game_id, game in all_games.items()):
        raise ValueError('Game sample counts disagree with stored rows')
    checks = {'teacher_calls': len(analyses), 'successful_teacher_calls': len(analyses),
              'known_node_calls': len(analyses), 'unknown_node_calls': 0,
              'requested_nodes': sum(call['requested_nodes'] for call in analyses),
              'reported_nodes': sum(call['reported_nodes'] for call in analyses),
              'teacher_wall_seconds': sum(call['wall_seconds'] for call in analyses),
              'teacher_call_wall_seconds': sum(call['call_wall_seconds'] for call in analyses),
              'unique_states': len(seen), 'excluded_states': len(excluded),
              'excluded_symmetry_classes': len(excluded_classes), 'reserved_state_keys': len(reserved),
              'generated_games': len(all_games), 'rollout_only_teacher_calls': len(analyses)-len(seen)}
    if any(not math.isclose(result[key], value, rel_tol=1e-12, abs_tol=1e-12)
           if isinstance(value, float) else result[key] != value for key, value in checks.items()):
        raise ValueError('Receipt call accounting disagrees with raw evidence')
    if sum(call['labelled_position'] for call in analyses) != len(seen):
        raise ValueError('Accepted teacher calls disagree with saved examples')
    return rows_by_split
