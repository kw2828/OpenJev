import hashlib
import json
import math
import random

import pytest

pytest.importorskip('chess')
import chess

from openjev.research.chess_spatial_regret import validate_regret


def fixture(directory):
    directory.mkdir()
    config = {'modes': ['cnn', 'recurrent', 'reconstruct', 'predict'], 'seeds': [17, 29, 43],
              'regret_nodes': 20000, 'regret_positions_per_split': 2, 'regret_selection_seed': 950001,
              'value_cp_scale': 600., 'mate_cp': 10000}
    policies = [f'{mode}-{seed}' for mode in config['modes'] for seed in config['seeds']]
    policies += ['greedy_material', 'training_move_frequency']
    rows, choices, selection, analyses, results = {}, {}, {}, [], []
    for split in ('dev', 'shift'):
        rows[split], choices[split] = [], {name: [] for name in policies}
        board = chess.Board()
        for index in range(3):
            legal = sorted(move.uci() for move in board.legal_moves)
            row = {'id': f'{split}-{index}', 'fen': board.fen(en_passant='fen'),
                   'target_uci': legal[0], 'game_id': 1}
            rows[split].append(row)
            for offset, policy in enumerate(policies):
                choices[split][policy].append({'id': row['id'], 'choice': legal[offset % 2],
                                               'game_id': 1, 'target': legal[0], 'correct': offset % 2 == 0,
                                               'target_nll': 1., 'target_probability': math.exp(-1), 'value': .3})
            board.push_uci(legal[0])
        selection[split] = sorted(random.Random(950001+(split == 'shift')).sample(range(3), 2))
        for index in selection[split]:
            row = rows[split][index]
            roots = sorted({predictions[index]['choice'] for predictions in choices[split].values()})
            lookup = {}
            for ordinal, root in enumerate([None]+roots):
                cp = [100, 150, 50][ordinal]
                call = {'split': split, 'id': row['id'], 'fen': row['fen'], 'root_move': root,
                        'pv_first': roots[0] if root is None else root, 'score_cp': cp, 'mate': None,
                        'bounded_score': math.tanh(cp/600), 'requested_nodes': 20000,
                        'reported_nodes': 20003, 'wall_seconds': .01}
                analyses.append(call)
                lookup[root] = call
            for policy in policies:
                root = choices[split][policy][index]['choice']
                results.append({'split': split, 'id': row['id'], 'policy': policy, 'choice': root,
                                'best_score': lookup[None]['bounded_score'],
                                'chosen_score': lookup[root]['bounded_score'],
                                'bounded_regret': lookup[None]['bounded_score']-lookup[root]['bounded_score'],
                                'cp_regret': lookup[None]['score_cp']-lookup[root]['score_cp']})
    means = {split: {policy: {f'mean_{field}': sum(row[field] for row in results
                                                 if row['split'] == split and row['policy'] == policy)/2
                              for field in ('bounded_regret', 'cp_regret')}
                    for policy in policies} for split in ('dev', 'shift')}
    receipt = {'status': 'completed', 'cost': {'calls': len(analyses), 'requested_nodes': 20000*len(analyses),
                                              'reported_nodes': 20003*len(analyses),
                                              'wall_seconds': .01*len(analyses)},
               'means': means, 'files': {}}
    files = {'selection.json': selection, 'analyses.jsonl': analyses, 'predictions.json': results,
             'completed.json': receipt}
    save(directory, files)
    return rows, choices, config, files


def save(directory, files):
    for name, value in files.items():
        if name == 'completed.json':
            continue
        text = ''.join(json.dumps(row)+'\n' for row in value) if name.endswith('.jsonl') else json.dumps(value)
        (directory/name).write_text(text)
        files['completed.json']['files'][name] = hashlib.sha256((directory/name).read_bytes()).hexdigest()
    (directory/'completed.json').write_text(json.dumps(files['completed.json']))


def test_complete_fourteen_policy_panel_and_negative_finite_search_difference(tmp_path):
    directory = tmp_path/'regret'
    rows, choices, config, files = fixture(directory)
    receipt = validate_regret(directory, rows, choices, config)
    assert receipt == files['completed.json']
    assert receipt['cost']['calls'] == 12
    assert len(files['predictions.json']) == 56
    assert receipt['means']['dev']['cnn-17']['mean_cp_regret'] == -50


@pytest.mark.parametrize('mutation', [
    'selection', 'duplicate_call', 'missing_call', 'fen', 'pv', 'root', 'cp', 'bounded',
    'requested_nodes', 'reported_nodes', 'wall', 'mate', 'cost', 'missing_result', 'duplicate_result',
    'choice', 'regret', 'means', 'mean_membership', 'files', 'status',
])
def test_rehashed_corrupt_evidence_is_rejected(tmp_path, mutation):
    directory = tmp_path/'regret'
    rows, choices, config, files = fixture(directory)
    analyses, predictions, receipt = files['analyses.jsonl'], files['predictions.json'], files['completed.json']
    if mutation == 'selection':
        files['selection.json']['dev'] = list(reversed(files['selection.json']['dev']))
    elif mutation == 'duplicate_call':
        analyses.append(analyses[0])
    elif mutation == 'missing_call':
        analyses.pop()
    elif mutation == 'fen':
        analyses[0]['fen'] = chess.STARTING_FEN+' '
    elif mutation == 'pv':
        analyses[0]['pv_first'] = 'a1a8'
    elif mutation == 'root':
        analyses[1]['pv_first'] = analyses[2]['pv_first']
    elif mutation == 'cp':
        analyses[0]['score_cp'] = float('nan')
    elif mutation == 'bounded':
        analyses[0]['bounded_score'] += .1
    elif mutation == 'requested_nodes':
        analyses[0]['requested_nodes'] = 2000
    elif mutation == 'reported_nodes':
        analyses[0]['reported_nodes'] = 0
    elif mutation == 'wall':
        analyses[0]['wall_seconds'] = -1
    elif mutation == 'mate':
        analyses[0]['mate'] = 3
    elif mutation == 'cost':
        receipt['cost']['calls'] -= 1
    elif mutation == 'missing_result':
        predictions.pop()
    elif mutation == 'duplicate_result':
        predictions.append(predictions[0])
    elif mutation == 'choice':
        predictions[0]['choice'] = predictions[1]['choice']
    elif mutation == 'regret':
        predictions[0]['bounded_regret'] = 0
    elif mutation == 'means':
        receipt['means']['dev']['cnn-17']['mean_cp_regret'] = 0
    elif mutation == 'mean_membership':
        receipt['means']['dev'].pop('cnn-17')
    elif mutation == 'status':
        receipt['status'] = 'partial'
    save(directory, files)
    if mutation == 'files':
        receipt['files'].pop('analyses.jsonl')
        (directory/'completed.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_regret(directory, rows, choices, config)


@pytest.mark.parametrize('mutation', ['missing_policy', 'extra_policy', 'missing_prediction',
                                    'wrong_identity', 'illegal_choice', 'probability'])
def test_incomplete_or_inconsistent_full_policy_predictions_are_rejected(tmp_path, mutation):
    directory = tmp_path/'regret'
    rows, choices, config, _ = fixture(directory)
    predictions = choices['dev']['cnn-17']
    if mutation == 'missing_policy':
        choices['dev'].pop('cnn-17')
    elif mutation == 'extra_policy':
        choices['dev']['extra'] = predictions
    elif mutation == 'missing_prediction':
        predictions.pop()
    elif mutation == 'wrong_identity':
        predictions[0]['id'] = 'other'
    elif mutation == 'illegal_choice':
        predictions[0]['choice'] = 'a1a8'
    elif mutation == 'probability':
        predictions[0]['target_probability'] = .9
    with pytest.raises(ValueError):
        validate_regret(directory, rows, choices, config)


def test_hash_failure_and_failed_marker_are_rejected(tmp_path):
    directory = tmp_path/'regret'
    rows, choices, config, _ = fixture(directory)
    (directory/'failed.json').write_text('{}')
    with pytest.raises(ValueError, match='complete'):
        validate_regret(directory, rows, choices, config)
    (directory/'failed.json').unlink()
    with (directory/'analyses.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash'):
        validate_regret(directory, rows, choices, config)
