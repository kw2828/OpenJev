# SPDX-License-Identifier: GPL-3.0-only
"""Inventory existing training-only behavior continuation labels, without inference.

This is retrospective data accounting, not a training protocol or confirmation
set. Native outcomes use the full recorded history. Nonterminal labels require
an accepted training root at the next ply of the same game and source.
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'chess-transition-value-inventory-v1'
SOURCES = {
    'runs/chess-spatial-v1/execution/data':
        '312d97c0ed6bcf1c9ecd267621c2b48532cd1f521a7237e16986e83d99d50bda',
    'runs/chess-capacity-v1/execution/data':
        '26f4975d62c48281592fb650dc74ed6b99ff00c2f946c410a989b992e738c058',
}
FILES = ('started.json', 'train.jsonl', 'games.jsonl', 'analyses.jsonl')
LABEL_FIELDS = ('target_uci', 'target_value', 'score_cp', 'mate')
COST_FIELDS = ('requested_nodes', 'reported_nodes', 'wall_seconds', 'call_wall_seconds')
SOURCE_COST_FIELDS = ('teacher_calls', 'successful_teacher_calls', 'requested_nodes',
                      'reported_nodes', 'teacher_wall_seconds', 'teacher_call_wall_seconds',
                      'known_node_calls', 'unknown_node_calls', 'rollout_only_teacher_calls',
                      'teacher_cost_scope')
LIMITS = ('Exposed historical development data only. One recorded behavior action per root; '
          'no alternative-action values, calibrated win probabilities, new teacher calls, '
          'training, evaluation or performance claim. Source teacher costs are inherited, '
          'not newly incurred; root/child role costs overlap and must not be added.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), f'Not a regular source file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank JSONL record')
            yield json.loads(line)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(row):
    require(type(row.get('game_id')) is int and row['game_id'] >= 0
            and type(row.get('ply')) is int and row['ply'] >= 0, 'Invalid game/ply identity')
    return row['game_id'], row['ply']


def state_key(board):
    return ' '.join(board.fen(en_passant='fen').split()[:4])


def check_call(call, board, config):
    require(call.get('status') == 'completed', 'Unsuccessful teacher call')
    require(call.get('fen') == board.fen(en_passant='fen')
            and call.get('state_key') == state_key(board), 'Teacher FEN or state identity mismatch')
    require(type(call.get('labelled_position')) is bool
            and type(call.get('excluded_position')) is bool, 'Malformed teacher admission flags')
    require(not (call['labelled_position'] and call['excluded_position']), 'Accepted excluded teacher root')
    require(chess.Move.from_uci(call['target_uci']) in board.legal_moves, 'Illegal teacher move')
    cp, mate, value = call.get('score_cp'), call.get('mate'), call.get('target_value')
    require(type(cp) is int and (mate is None or type(mate) is int), 'Malformed teacher score')
    if mate is not None:
        expected_cp = config['mate_cp'] - mate if mate > 0 else -config['mate_cp'] - mate
        require(cp == expected_cp, 'Mate to centipawn score mismatch')
    require(finite(value) and -1 <= value <= 1
            and math.isclose(value, math.tanh(cp/config['value_cp_scale']),
                             rel_tol=0, abs_tol=1e-12), 'Teacher value scaling mismatch')
    require(type(call.get('requested_nodes')) is int
            and call['requested_nodes'] == config['teacher_nodes'], 'Teacher node request mismatch')
    require(type(call.get('reported_nodes')) is int and call['reported_nodes'] >= 0,
            'Malformed reported node cost')
    for key in ('wall_seconds', 'call_wall_seconds'):
        require(finite(call.get(key)) and call[key] >= 0, 'Malformed teacher wall cost')


def check_row(row, call, board, game, action):
    require(row.get('split') == 'train' and row.get('game_seed') == game['seed']
            and type(row.get('game_seed')) is int, 'Root split or game seed mismatch')
    require(row.get('fen') == board.fen(en_passant='fen')
            and row.get('state_key') == state_key(board), 'Root FEN or state identity mismatch')
    require(call['labelled_position'] and not call['excluded_position'], 'Root not accepted for training')
    require(all(row.get(k) == call[k] for k in LABEL_FIELDS), 'Root label differs from teacher record')
    require(row.get('behavior_uci') == action['uci'], 'Behavior move mismatch')
    successor = board.copy(stack=True)
    successor.push_uci(action['uci'])
    require(row.get('next_fen') == successor.fen(en_passant='fen'), 'Successor FEN mismatch')


def costs(calls):
    calls = list(calls)
    return {'calls': len(calls), **{key: sum(c[key] for c in calls) for key in COST_FIELDS}}


def teacher_reference(call):
    return {key: call[key] for key in ('split', 'game_id', 'ply', 'fen', 'target_uci',
                                      'target_value', 'score_cp', 'mate', *COST_FIELDS)}


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0, 'mean': None, 'quantiles': None}
    def quantile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {'count': len(values), 'mean': sum(values)/len(values),
            'quantiles': {str(q): quantile(q) for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)}}


def consistency(records):
    result = {}
    for same in (True, False):
        pairs = [r for r in records if r['label_kind'] == 'accepted_train_child_teacher'
                 and r['behavior_equals_teacher'] == same]
        differences = [r['continuation_value'] - r['root_teacher']['target_value'] for r in pairs]
        result['behavior_equals_teacher' if same else 'behavior_other'] = {
            'absolute_difference': distribution([abs(x) for x in differences]),
            'signed_difference': distribution(differences),
            'continuation_exceeds_root_count': sum(x > 0 for x in differences),
        }
    result['interpretation'] = ('Descriptive consistency of different finite teacher searches, with all '
                                'eligible pairs retained. Not exact Bellman residuals, regret, '
                                'calibrated probabilities or a quality-based admission filter.')
    return result


def continuation(row, child, next_call, successor, root_turn):
    """Return a value only with native terminal proof or an admitted next root."""
    outcome = successor.outcome(claim_draw=False)
    if outcome is not None:
        require(child is None and next_call is None, 'Terminal successor has a later visited root')
        value = 0.0 if outcome.winner is None else (1.0 if outcome.winner == root_turn else -1.0)
        return {'status': 'accepted', 'label_kind': 'native_terminal', 'continuation_value': value,
                'termination': outcome.termination.name.lower(), 'child_teacher': None,
                'child_id': None, 'reason': None}
    if child is None:
        require(next_call is None or not next_call['labelled_position'], 'Accepted child row is missing')
        reason = ('child_unobserved' if next_call is None else
                  'child_excluded' if next_call['excluded_position'] else 'child_not_accepted')
        return {'status': 'skipped', 'label_kind': None, 'continuation_value': None,
                'termination': None, 'child_teacher': None, 'child_id': None, 'reason': reason}
    require(next_call is not None and next_call['labelled_position']
            and not next_call['excluded_position'], 'Child not admitted for training')
    require(identity(child) == (row['game_id'], row['ply'] + 1)
            and identity(next_call) == identity(child)
            and child['split'] == next_call['split'] == 'train'
            and child['game_seed'] == row['game_seed'], 'Child source game/ply identity mismatch')
    require(child['fen'] == next_call['fen'] == row['next_fen'] == successor.fen(en_passant='fen'),
            'Child FEN identity mismatch')
    require(all(child[k] == next_call[k] for k in LABEL_FIELDS), 'Child label differs from teacher record')
    return {'status': 'accepted', 'label_kind': 'accepted_train_child_teacher',
            'continuation_value': -child['target_value'], 'termination': None,
            'child_teacher': teacher_reference(next_call), 'child_id': child['id'], 'reason': None}


def inventory_source(directory, expected_receipt, source_id):
    """Authenticate one fixed source, then replay its training histories only."""
    require(sha(directory/'completed.json') == expected_receipt, 'Source receipt hash mismatch')
    receipt = read(directory/'completed.json')
    require(receipt.get('status') == 'completed' and not (directory/'failed.json').exists(),
            'Source not successfully completed')
    hashes = {'completed.json': expected_receipt}
    for name in FILES:
        require(sha(directory/name) == receipt['files'][name], f'Source file hash mismatch: {name}')
        hashes[name] = receipt['files'][name]
    config = read(directory/'started.json')['config']
    require(finite(config['value_cp_scale']) and config['value_cp_scale'] > 0
            and type(config['mate_cp']) is int and config['mate_cp'] > 0
            and type(config['teacher_nodes']) is int and config['teacher_nodes'] > 0,
            'Invalid teacher scale or budget')
    splits = [s for s in config['splits'] if s['name'] == 'train']
    require(len(splits) == 1 and config['splits'][0]['name'] == 'train', 'Unexpected training split')
    split = splits[0]
    rows, positions, row_ids = [], {}, set()
    for index, row in enumerate(jsonl(directory/'train.jsonl')):
        require(row.get('split') == 'train' and row.get('id') == f'train-{index:05d}',
                'Incorrect training row order or split')
        key = identity(row)
        require(key not in positions and row['id'] not in row_ids, 'Duplicate accepted root')
        rows.append(row)
        positions[key] = row
        row_ids.add(row['id'])
    require(len(rows) == split['examples'] == receipt['counts']['train'], 'Training root count mismatch')
    games = {}
    for game in jsonl(directory/'games.jsonl'):
        if game['split'] != 'train':
            continue  # No held-out move history is reconstructed.
        gid = game.get('game_id')
        require(type(gid) is int and 0 <= gid < split['game_cap'] and gid not in games,
                'Duplicate or invalid training game')
        require(type(game.get('seed')) is int and game['seed'] == split['seed_base'] + gid,
                'Source game seed mismatch')
        require(0 < len(game['actions']) <= split['max_plies'], 'Game ply budget mismatch')
        games[gid] = game
    calls = {}
    for call in jsonl(directory/'analyses.jsonl'):
        if call['split'] != 'train':
            continue  # No held-out labels are used, joined, or copied to outputs.
        key = identity(call)
        require(key not in calls and key[0] in games, 'Duplicate or unknown teacher position')
        calls[key] = call
    require(len(calls) == sum(len(g['actions']) for g in games.values()), 'Training call coverage mismatch')
    require(all(key[0] in games for key in positions), 'Accepted root belongs to another game/source')
    records, visited, used_children = {}, set(), set()
    for gid, game in games.items():
        board = chess.Board()
        sampled = 0
        for ply, action in enumerate(game['actions']):
            key = gid, ply
            require(board.outcome(claim_draw=False) is None, 'Action after native terminal outcome')
            require(key in calls, 'Missing root teacher call')
            call = calls[key]
            check_call(call, board, config)
            row = positions.get(key)
            require(call['labelled_position'] == (row is not None), 'Call acceptance differs from train rows')
            move = chess.Move.from_uci(action['uci'])
            require(move in board.legal_moves and action['source'] in ('random', 'teacher'),
                    'Invalid recorded behavior action')
            require(action['source'] != 'teacher' or action['uci'] == call['target_uci'],
                    'Teacher-sourced action differs from teacher target')
            if row is not None:
                check_row(row, call, board, game, action)
                successor = board.copy(stack=True)
                successor.push(move)
                next_key = gid, ply + 1
                label = continuation(row, positions.get(next_key), calls.get(next_key), successor, board.turn)
                if label['label_kind'] == 'accepted_train_child_teacher':
                    used_children.add(next_key)
                records[key] = {
                    'source': source_id, 'source_receipt_sha256': expected_receipt,
                    'id': row['id'], 'split': 'train', 'game_id': gid, 'game_seed': game['seed'],
                    'ply': ply, 'fen': row['fen'], 'next_fen': row['next_fen'],
                    'behavior_uci': row['behavior_uci'], 'behavior_source': action['source'],
                    'behavior_equals_teacher': row['behavior_uci'] == row['target_uci'],
                    'root_player': 'white' if board.turn == chess.WHITE else 'black',
                    'root_teacher': teacher_reference(call), **label,
                }
                sampled += 1
            visited.add(key)
            board.push(move)
        require(board.fen(en_passant='fen') == game['final_fen'], 'Final game FEN mismatch')
        require(sampled == game['sampled_positions'], 'Game accepted-root count mismatch')
        stop = game['stop']
        require(stop in ('terminal', 'ply_cap', 'quota'), 'Invalid game stop reason')
        if stop == 'terminal':
            require(board.outcome(claim_draw=False) is not None, 'Terminal marker without native outcome')
        elif stop == 'ply_cap':
            require(len(game['actions']) == split['max_plies'], 'Premature ply cap')
        else:
            require(gid == max(games) and (gid, len(game['actions']) - 1) in positions,
                    'Quota stop is not final accepted training root')
    require(visited == set(calls) and len(records) == len(rows), 'Replay identity coverage mismatch')
    ordered = [records[identity(row)] for row in rows]
    categories = Counter((r['label_kind'] if r['status'] == 'accepted' else r['reason']) for r in ordered)
    accepted = [r for r in ordered if r['status'] == 'accepted']
    root_keys = set(positions)
    referenced = root_keys | used_children
    stats = {
        'source': source_id, 'consumed_files': hashes, 'train_roots': len(rows), 'train_games': len(games),
        'accepted': len(accepted), 'skipped': len(rows) - len(accepted), 'categories': dict(categories),
        'behavior_source_all': dict(Counter(r['behavior_source'] for r in ordered)),
        'behavior_source_accepted': dict(Counter(r['behavior_source'] for r in accepted)),
        'behavior_match_all': dict(Counter('teacher' if r['behavior_equals_teacher'] else 'other' for r in ordered)),
        'behavior_match_accepted': dict(Counter('teacher' if r['behavior_equals_teacher'] else 'other' for r in accepted)),
        'terminal_outcomes': dict(Counter(str(r['continuation_value']) for r in accepted
                                          if r['label_kind'] == 'native_terminal')),
        'teacher_configuration': {k: config[k] for k in ('teacher_nodes', 'teacher_threads',
                                                        'teacher_hash_mb', 'value_cp_scale', 'mate_cp')},
        'inherited_all_source_costs': {k: receipt[k] for k in SOURCE_COST_FIELDS},
        'replayed_training_call_costs': costs(calls.values()),
        'referenced_unique_call_costs': costs(calls[k] for k in sorted(referenced)),
        'root_role_costs': costs(calls[k] for k in sorted(root_keys)),
        'child_role_costs': costs(calls[k] for k in sorted(used_children)),
        'new_teacher_calls': 0,
        'nonterminal_consistency': consistency(ordered),
    }
    require(all(sha(directory/name) == digest for name, digest in hashes.items()),
            'Source changed during inventory')
    return ordered, stats


def run(out):
    begin = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'started.json', {'version': VERSION, 'unix': time.time(), 'sources': SOURCES,
                              'limits': LIMITS, 'python_chess_version': chess.__version__})
    try:
        summaries = []
        with (out/'records.jsonl').open('x') as stream:
            for source, expected in SOURCES.items():
                records, stats = inventory_source(ROOT/source, expected, source)
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                summaries.append(stats)
                print(json.dumps({k: stats[k] for k in ('source', 'train_roots', 'accepted', 'skipped', 'categories')}),
                      flush=True)
        summary = {'version': VERSION, 'status': 'completed', 'sources': summaries,
                   'train_roots': sum(s['train_roots'] for s in summaries),
                   'accepted': sum(s['accepted'] for s in summaries),
                   'skipped': sum(s['skipped'] for s in summaries),
                   'new_teacher_calls': 0, 'training_updates': 0, 'wall_seconds': time.monotonic()-begin,
                   'limits': LIMITS}
        write(out/'summary.json', summary)
        text = ('# Saved training continuation inventory\n\n'
                f"{summary['accepted']:,} usable behavior continuations from {summary['train_roots']:,} "
                f"accepted training roots; {summary['skipped']:,} explicitly skipped.\n\n"
                'Every records.jsonl line represents one accepted historical TRAIN root. '
                'Skipped rows have a null continuation value, never an invented zero. '
                'Nonterminal values negate the accepted next TRAIN root teacher value in the same '
                'source and game; terminal values use native chess outcomes after replaying the '
                'complete recorded move history with claim_draw=False.\n\n'
                'Teacher values are tanh(centipawns / 600), not calibrated win probabilities. '
                'Random-sourced moves can coincide with teacher choices; both fields are retained. '
                'All source-file hashes and inherited call costs are in summary.json. Role costs overlap. '
                'No dev/shift label is used as a target.\n\n' + LIMITS + '\n')
        with (out/'README.md').open('x') as stream:
            stream.write(text)
        code = ('scripts/chess_transition_value_inventory.py', 'tests/test_chess_transition_value_inventory.py')
        write(out/'receipt.json', {'version': VERSION, 'status': 'completed',
                                  'sources': SOURCES, 'code': {p: sha(ROOT/p) for p in code},
                                  'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
                                  'new_teacher_calls': 0, 'training_updates': 0, 'limits': LIMITS})
        print(json.dumps({k: summary[k] for k in ('status', 'train_roots', 'accepted', 'skipped', 'wall_seconds')}),
              flush=True)
    except BaseException as error:
        write(out/'failed.json', {'error': repr(error), 'wall_seconds': time.monotonic()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
