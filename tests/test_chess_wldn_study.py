# SPDX-License-Identifier: GPL-3.0-only
import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_wldn_study as study
from openjev.research.chess_candidate import CandidateChess, encode_batch


def values():
    return {f'{arm}-{s}': {split: {'agreement': .34 if arm == 'contrast' else .30} for split in ('dev', 'shift')}
            for arm in ('contrast', 'wldn') for s in (97, 109, 127)}


def test_both_panels_mean_threshold_and_seed_floor_are_required():
    metrics = values(); assert all(r['passed'] for r in study.gate(metrics))
    for s in (97, 109, 127): metrics[f'wldn-{s}']['shift']['agreement'] = .335
    assert study.gate(metrics)[0]['passed'] and not study.gate(metrics)[1]['passed']
    metrics = values(); metrics['contrast-97']['dev']['agreement'] = .29
    metrics['contrast-109']['dev']['agreement'] = .43
    assert study.gate(metrics)[0]['mean_gain'] > .01 and not study.gate(metrics)[0]['passed']


def test_budget_matches_existing_comparison_without_waiting_for_outcomes():
    p = study.PROTOCOL
    assert p['epochs']*p['training_examples']//p['batch_size'] == p['updates_per_fit'] == 1536
    assert p['fits'] == 3 and p['total_updates'] == 4608 and p['parameters'] == 16638
    assert 'all8 original contrast checks' in p['comparison']
    assert 'recomputed child node features' in p['information']


def test_complete_native_decision_matches_batched_head_with_special_moves():
    torch.set_num_threads(2)
    fens = [chess.STARTING_FEN, '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2',
            'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', '4k3/P7/8/8/8/8/8/4K3 w - - 0 1']
    fens += [chess.Board(f).mirror().fen(en_passant='fen') for f in fens]
    boards = [chess.Board(f) for f in fens]; inputs, menus = encode_batch(boards, 'direct')
    model = CandidateChess('direct', seed=37, width=32).eval().requires_grad_(False)
    head = study.ChessWLDNHead(seed=1137)
    with torch.no_grad():
        head.output.weight.fill_(.1)
        base, _, hidden = model(**inputs); nodes = hidden.flatten(2).transpose(1, 2)
        features = study.contrast.prior.candidate_features(model, nodes, inputs['candidates'])
        graphs = study.contrast.candidate_graphs(boards)
        scores = head(nodes, features, inputs['candidates'], inputs['legal_mask'], base, graphs['root'], graphs['children'][graphs['mask']])
        choices = scores.argmax(-1).tolist()
    for i, fen in enumerate(fens): assert study.decision(model, head, fen) == menus[i][choices[i]]
