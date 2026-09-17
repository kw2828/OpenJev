"""Independently verify the complete fixed-panel secondary engine assessment."""

import hashlib
import json
import math
import random
from pathlib import Path

import chess
import chess.engine


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _equal(actual, expected, message):
    if not _finite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(message)


def validate_regret(directory, rows, choices, config):
    """Return the unchanged receipt after auditing raw calls and derived results.

    ``choices[split][policy]`` is the full prediction list from the validated
    fit or baseline, not the smaller secondary panel. Exact policy membership is
    determined from config modes/seeds plus the two deterministic baselines.
    Negative finite-search score differences are allowed and retained.
    """
    directory = Path(directory)
    receipt = json.loads((directory/'completed.json').read_text())
    if receipt['status'] != 'completed' or (directory/'failed.json').exists():
        raise ValueError('Secondary engine assessment is not complete')
    required = {'selection.json', 'analyses.jsonl', 'predictions.json'}
    if set(receipt['files']) != required:
        raise ValueError('Regret receipt must bind all and only required evidence files')
    for name, digest in receipt['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Regret evidence hash mismatch')
    if (type(config['regret_nodes']) is not int or config['regret_nodes'] <= 0 or
            type(config['regret_positions_per_split']) is not int or config['regret_positions_per_split'] <= 0):
        raise ValueError('Regret call and panel budgets must be positive integers')
    if not _finite(config['value_cp_scale']) or config['value_cp_scale'] <= 0:
        raise ValueError('Regret score scale must be positive and finite')
    splits = ('dev', 'shift')
    policies = {f'{mode}-{seed}' for mode in config['modes'] for seed in config['seeds']}
    policies.update(('greedy_material', 'training_move_frequency'))
    if len(policies) != len(config['modes'])*len(config['seeds'])+2 or set(choices) != set(splits):
        raise ValueError('Invalid configured policy identities or evaluation split membership')
    selected = json.loads((directory/'selection.json').read_text())
    expected_selection = {}
    indexed = {}
    for split in splits:
        if not rows[split] or len({row['id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('Evaluation rows must have unique IDs and be nonempty')
        if set(choices[split]) != policies:
            raise ValueError('Incomplete or extra regret policy membership')
        expected_selection[split] = sorted(random.Random(config['regret_selection_seed']+(split == 'shift')).sample(
            range(len(rows[split])), min(len(rows[split]), config['regret_positions_per_split'])))
        boards = [chess.Board(row['fen']) for row in rows[split]]
        if any(not board.is_valid() for board in boards):
            raise ValueError('Invalid evaluation board')
        for policy, predictions in choices[split].items():
            if len(predictions) != len(rows[split]):
                raise ValueError('Policy prediction list does not cover the complete evaluation split')
            for row, prediction, board in zip(rows[split], predictions, boards, strict=True):
                if prediction['id'] != row['id']:
                    raise ValueError('Policy prediction ordering or position identity mismatch')
                if chess.Move.from_uci(prediction['choice']) not in board.legal_moves:
                    raise ValueError('Policy prediction contains an illegal choice')
                for key, source in (('game_id', 'game_id'), ('target', 'target_uci')):
                    if key in prediction and prediction[key] != row[source]:
                        raise ValueError('Policy prediction metadata disagrees with its evaluation row')
                if 'correct' in prediction and prediction['correct'] != (prediction['choice'] == row['target_uci']):
                    raise ValueError('Policy prediction correctness flag is inconsistent')
                if 'target_nll' in prediction:
                    nll = prediction['target_nll']
                    if not _finite(nll) or nll < 0:
                        raise ValueError('Policy prediction NLL is invalid')
                    _equal(prediction['target_probability'], math.exp(-nll),
                           'Policy probability does not agree with NLL')
                if 'value' in prediction and (not _finite(prediction['value']) or
                                              not -1 <= prediction['value'] <= 1):
                    raise ValueError('Policy value prediction is invalid')
        for index in expected_selection[split]:
            row = rows[split][index]
            indexed[split, row['id']] = (index, row, boards[index])
    if selected != expected_selection:
        raise ValueError('Secondary panel differs from the frozen seeded selection')
    analyses = [json.loads(line) for line in (directory/'analyses.jsonl').read_text().splitlines()]
    expected_calls = set()
    expected_results = set()
    for (split, identifier), (index, _, _) in indexed.items():
        expected_calls.add((split, identifier, None))
        for policy, predictions in choices[split].items():
            expected_calls.add((split, identifier, predictions[index]['choice']))
            expected_results.add((split, identifier, policy))
    lookup = {}
    costs = {'calls': 0, 'requested_nodes': 0, 'reported_nodes': 0, 'wall_seconds': 0.}
    for call in analyses:
        key = call['split'], call['id'], call['root_move']
        if key not in expected_calls or key in lookup:
            raise ValueError('Extra, duplicate or unidentified secondary engine call')
        _, row, board = indexed[key[:2]]
        if call['fen'] != row['fen']:
            raise ValueError('Secondary engine call FEN differs from its evaluation position')
        root, first = call['root_move'], chess.Move.from_uci(call['pv_first'])
        if first not in board.legal_moves or (root is not None and first.uci() != root):
            raise ValueError('Secondary engine PV is illegal or does not match its restricted root')
        if type(call['score_cp']) is not int or not _finite(call['score_cp']):
            raise ValueError('Secondary engine centipawn score must be a finite integer')
        if call['mate'] is not None:
            if type(call['mate']) is not int:
                raise ValueError('Secondary engine mate distance must be an integer or null')
            if call['score_cp'] != chess.engine.Mate(call['mate']).score(mate_score=config['mate_cp']):
                raise ValueError('Secondary engine mate and centipawn scores disagree')
        _equal(call['bounded_score'], math.tanh(call['score_cp']/config['value_cp_scale']),
               'Secondary engine bounded score is inconsistent with its centipawn score')
        if type(call['requested_nodes']) is not int or call['requested_nodes'] != config['regret_nodes']:
            raise ValueError('Secondary engine call used an incorrect requested node budget')
        if type(call['reported_nodes']) is not int or call['reported_nodes'] <= 0:
            raise ValueError('Secondary engine reported node count must be positive')
        if not _finite(call['wall_seconds']) or call['wall_seconds'] < 0:
            raise ValueError('Secondary engine wall time must be finite and nonnegative')
        lookup[key] = call
        costs['calls'] += 1
        for field in ('requested_nodes', 'reported_nodes', 'wall_seconds'):
            costs[field] += call[field]
    if set(lookup) != expected_calls:
        raise ValueError('Missing unrestricted or unique chosen-root engine call')
    if set(receipt['cost']) != set(costs):
        raise ValueError('Regret receipt cost fields are incomplete')
    for field, expected in costs.items():
        if field == 'wall_seconds':
            _equal(receipt['cost'][field], expected, 'Regret call wall time total is inconsistent')
        elif type(receipt['cost'][field]) is not int or receipt['cost'][field] != expected:
            raise ValueError('Regret call or node total is inconsistent')
    results = json.loads((directory/'predictions.json').read_text())
    seen_results = set()
    aggregates = {(split, policy): [] for split in splits for policy in policies}
    for result in results:
        key = result['split'], result['id'], result['policy']
        if key not in expected_results or key in seen_results:
            raise ValueError('Extra, duplicate or unidentified regret prediction')
        seen_results.add(key)
        index, _, _ = indexed[key[:2]]
        expected_choice = choices[key[0]][key[2]][index]['choice']
        if result['choice'] != expected_choice:
            raise ValueError('Regret prediction choice differs from the saved policy prediction')
        best, chosen = lookup[*key[:2], None], lookup[*key[:2], expected_choice]
        expected = {'best_score': best['bounded_score'], 'chosen_score': chosen['bounded_score'],
                    'bounded_regret': best['bounded_score']-chosen['bounded_score'],
                    'cp_regret': best['score_cp']-chosen['score_cp']}
        for field, value in expected.items():
            _equal(result[field], value, 'Regret prediction arithmetic is inconsistent')
        aggregates[key[0], key[2]].append(expected)
    if seen_results != expected_results:
        raise ValueError('Missing policy-by-panel regret prediction')
    if set(receipt['means']) != set(splits):
        raise ValueError('Regret mean split membership is incomplete')
    for split in splits:
        if set(receipt['means'][split]) != policies:
            raise ValueError('Regret mean policy membership is incomplete')
        for policy in policies:
            values = aggregates[split, policy]
            mean = receipt['means'][split][policy]
            if set(mean) != {'mean_bounded_regret', 'mean_cp_regret'}:
                raise ValueError('Regret mean fields are incomplete')
            for field in ('bounded_regret', 'cp_regret'):
                _equal(mean['mean_'+field], math.fsum(row[field] for row in values)/len(values),
                       'Regret mean disagrees with recomputed prediction arithmetic')
    return receipt
