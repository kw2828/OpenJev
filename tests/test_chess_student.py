import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

pytest.importorskip('chess')
import chess

from openjev.research.chess_student import (
    DEPTH,
    INPUT_SIZE,
    MODES,
    MOVE_TO_INDEX,
    MOVE_VOCAB_SHA256,
    UCI_MOVES,
    ChessStudent,
    circuit_topology,
    encode_board,
    legal_mask,
    state_key,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chess_student_training', ROOT/'scripts/train_chess_student.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def tiny_config(**overrides):
    return {**study.PROTOCOL, 'train_examples': 12, 'dev_examples': 6,
            'train_game_cap': 8, 'dev_game_cap': 4, 'game_max_plies': 12,
            'random_move_probability': 1., **overrides}


def fake_label(board):
    target = min(move.uci() for move in board.legal_moves)
    return {'target_uci': target, 'target_value': float(np.tanh(100/600)), 'score_cp': 100, 'mate': None,
            'requested_nodes': 2000, 'reported_nodes': 2001, 'wall_seconds': .001}


def test_geometric_vocab_is_complete_sorted_unique_and_contains_special_legal_moves():
    assert len(UCI_MOVES) == len(set(UCI_MOVES)) == 1968
    assert list(UCI_MOVES) == sorted(UCI_MOVES) and len(MOVE_VOCAB_SHA256) == 64
    for fen in [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '7k/P7/8/8/8/8/8/7K w - - 0 1', '7k/8/8/8/8/8/p7/7K b - - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2']:
        board = chess.Board(fen)
        assert all(move.uci() in MOVE_TO_INDEX for move in board.legal_moves)
        assert legal_mask(board).sum() == board.legal_moves.count()
    assert '0000' not in MOVE_TO_INDEX


def test_numeric_encoder_contains_fen_state_but_no_teacher_or_policy_attributes():
    board = chess.Board()
    first = encode_board(board)
    assert first.shape == (INPUT_SIZE,) and first.dtype == np.float32 and INPUT_SIZE == 841
    assert first[:768].sum() == 32 and first[768:770].tolist() == [1., 0.]
    assert first[770:774].tolist() == [1., 1., 1., 1.]
    assert first[774:839].sum() == 1 and first[838] == 1
    board.teacher_target = 'e2e4'
    board.engine_score = 9000
    np.testing.assert_array_equal(first, encode_board(board))
    board.turn = chess.BLACK
    assert encode_board(board)[768:770].tolist() == [0., 1.]
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    assert encode_board(board)[770:774].sum() == 0
    ep = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2')
    assert encode_board(ep)[774+chess.D6] == 1
    without_ep = ep.copy()
    without_ep.ep_square = None
    assert not np.array_equal(encode_board(ep), encode_board(without_ep))


def test_dedup_key_ignores_counters_but_keeps_side_castling_and_ep():
    board, altered = chess.Board(), chess.Board()
    altered.halfmove_clock, altered.fullmove_number = 15, 30
    assert state_key(board) == state_key(altered)
    assert not np.array_equal(encode_board(board), encode_board(altered))
    for name, value in [('turn', chess.BLACK), ('castling_rights', chess.BB_EMPTY), ('ep_square', chess.E3)]:
        changed = board.copy()
        setattr(changed, name, value)
        assert state_key(changed) != state_key(board)


def test_rewired_control_changes_layout_but_preserves_signed_degrees_and_parameter_count():
    structured, rewired = circuit_topology(), circuit_topology(True)
    assert not torch.equal(structured, rewired)
    assert torch.equal(rewired, circuit_topology(True))
    assert structured.count_nonzero() == rewired.count_nonzero() == 352
    assert not structured.diag().any() and not rewired.diag().any()
    for sign in (-1, 1):
        for axis in (0, 1):
            assert torch.equal((structured == sign).sum(axis), (rewired == sign).sum(axis))
    for target, source in zip(*torch.where(structured != 0), strict=True):
        assert ((source < 16 and 16 <= target < 40) or
                (16 <= source < 40 and 40 <= target < 56) or
                (40 <= source < 56 and 40 <= target < 64))
    assert ChessStudent('circuit').parameter_count() == ChessStudent('rewired').parameter_count()
    assert ChessStudent('gru').parameter_count() != ChessStudent('circuit').parameter_count()


def test_shared_encoder_and_heads_have_identical_initial_weights_for_each_seed():
    for seed in (17, 29, 43):
        models = [ChessStudent(mode, seed) for mode in MODES]
        for name in ('encoder', 'policy_head', 'value_head'):
            state = getattr(models[0], name).state_dict()
            for model in models[1:]:
                for key, tensor in getattr(model, name).state_dict().items():
                    assert torch.equal(state[key], tensor)
        assert torch.equal(models[1].core.magnitude, models[2].core.magnitude)


@pytest.mark.parametrize('mode', MODES)
def test_untrained_forward_backward_smoke_is_finite_legal_and_does_not_update_weights(mode):
    torch.set_num_threads(2)
    model = ChessStudent(mode, 17)
    board = chess.Board()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    observations = torch.from_numpy(encode_board(board)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(board)).unsqueeze(0)
    logits, value = model(observations, mask)
    assert logits.shape == (1, 1968) and value.shape == (1,) and DEPTH == 4
    assert torch.isneginf(logits[~mask]).all()
    target = torch.tensor([MOVE_TO_INDEX['e2e4']])
    loss = F.cross_entropy(logits, target)+.5*value.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.parameters())
    for key, tensor in model.state_dict().items():
        assert torch.equal(initial[key], tensor)
    decision = model.choose(board)
    assert chess.Move.from_uci(decision['choice']) in board.legal_moves
    assert set(decision['probabilities']) == {move.uci() for move in board.legal_moves}
    assert sum(decision['probabilities'].values()) == pytest.approx(1.)
    assert decision['depth'] == 4 and -1 <= decision['value'] <= 1


def test_generation_is_exact_game_separated_and_state_deduplicated(tmp_path):
    config = tiny_config()
    receipt = study.generate_data(tmp_path/'data', fake_label, config=config)
    rows = study.validate_data(tmp_path/'data', config=config)
    assert receipt['counts'] == {'train': 12, 'dev': 6} and receipt['unique_states'] == 18
    assert {r['game_id'] for r in rows['train']}.isdisjoint({r['game_id'] for r in rows['dev']})
    assert {r['state_key'] for r in rows['train']}.isdisjoint({r['state_key'] for r in rows['dev']})
    assert receipt['requested_nodes'] == receipt['teacher_calls']*2000
    assert receipt['reported_nodes'] == receipt['teacher_calls']*2001
    second = study.generate_data(tmp_path/'same', fake_label, config=config)
    assert receipt['files'] == second['files']
    data = study.tensors(rows['train'])
    assert data['observations'].shape == (12, INPUT_SIZE)
    assert data['mask'].gather(1, data['targets'].unsqueeze(1)).all()


def test_insufficient_fixed_game_cap_leaves_durable_failure_and_partial_data(tmp_path):
    config = tiny_config(train_examples=1000, train_game_cap=1, game_max_plies=2)
    with pytest.raises(ValueError, match='fixed game cap'):
        study.generate_data(tmp_path/'data', fake_label, config=config)
    failure = json.loads((tmp_path/'data/failed.json').read_text())
    assert failure['status'] == 'failed' and failure['unique_states'] == 2
    assert len(study.jsonl(tmp_path/'data/train.jsonl')) == 2
    assert not (tmp_path/'data/completed.json').exists()
    with pytest.raises(FileExistsError):
        study.generate_data(tmp_path/'data', fake_label, config=config)


def test_teacher_failure_is_not_retried_or_replaced(tmp_path):
    calls = []

    def broken(board):
        calls.append(board.fen())
        raise RuntimeError('teacher failure')

    with pytest.raises(RuntimeError, match='teacher failure'):
        study.generate_data(tmp_path/'data', broken, config=tiny_config())
    assert len(calls) == 1 and (tmp_path/'data/failed.json').exists()
    assert study.jsonl(tmp_path/'data/train.jsonl') == []


def test_tampered_data_is_rejected_before_training(tmp_path):
    config = tiny_config()
    study.generate_data(tmp_path/'data', fake_label, config=config)
    path = tmp_path/'data/dev.jsonl'
    path.write_text(path.read_text().replace('dev-00000', 'train-00000'))
    with pytest.raises(ValueError, match='hash mismatch'):
        study.validate_data(tmp_path/'data', config=config)


def test_duplicate_state_is_rejected_even_with_updated_file_receipt(tmp_path):
    config = tiny_config()
    study.generate_data(tmp_path/'data', fake_label, config=config)
    train = study.jsonl(tmp_path/'data/train.jsonl')
    dev = study.jsonl(tmp_path/'data/dev.jsonl')
    dev[0].update({key: train[0][key] for key in ('fen', 'state_key', 'target_uci', 'target_value', 'score_cp')})
    path = tmp_path/'data/dev.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in dev))
    receipt_path = tmp_path/'data/completed.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['files']['dev.jsonl'] = study.sha(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='duplicated state'):
        study.validate_data(tmp_path/'data', config=config)


@pytest.mark.parametrize('mode', MODES)
def test_two_update_fake_teacher_fit_saves_final_checkpoint_and_exact_budget(mode, tmp_path, monkeypatch):
    config = tiny_config(epochs=1, batch_size=6)
    monkeypatch.setattr(study, 'PROTOCOL', config)
    study.generate_data(tmp_path/'data', fake_label, config=config)
    rows = study.validate_data(tmp_path/'data', config=config)
    train, dev = study.tensors(rows['train']), study.tensors(rows['dev'])
    receipt = study.fit(mode, 17, train, dev, rows['dev'], tmp_path/'fit', 'test-plan', 'test-data')
    assert receipt['updates'] == 2 and receipt['examples_seen'] == 12
    assert len(study.jsonl(tmp_path/'fit/learning.jsonl')) == 2
    loaded = ChessStudent.load(tmp_path/'fit/weights.pt', expected_plan_sha256='test-plan')
    assert loaded.mode == mode and loaded.seed == 17
    assert not torch.equal(loaded.policy_head.weight, ChessStudent(mode, 17).policy_head.weight)
    result = json.loads((tmp_path/'fit/evaluation.json').read_text())
    assert result['metrics']['examples'] == 6
    recomputed = study.evaluate(loaded, dev, rows['dev'])
    assert recomputed['predictions'] == result['predictions']
    with pytest.raises(ValueError, match='plan hash'):
        ChessStudent.load(tmp_path/'fit/weights.pt', expected_plan_sha256='wrong-plan')


def test_simple_baselines_use_train_frequency_and_exact_uniform_expectation():
    train = [{'target_uci': 'e2e4'}, {'target_uci': 'e2e4'}, {'target_uci': 'd2d4'}]
    dev = [{'id': 'heldout', 'fen': chess.STARTING_FEN, 'target_uci': 'd2d4'}]
    result = study.simple_baselines(train, dev)
    assert result['uniform_random']['expected_top1_teacher_agreement'] == .05
    frequency = result['training_move_frequency']
    assert frequency['top1_teacher_agreement'] == 0
    assert frequency['predictions'][0]['choice'] == 'e2e4'  # Never chosen from held-out target frequencies.


def test_continuation_gate_requires_all_four_predeclared_comparisons():
    fits = [{'mode': mode, 'seed': seed, 'top1_teacher_agreement': .6 if mode == 'circuit' else .5,
             'value_mae': .2} for mode in MODES for seed in study.PROTOCOL['seeds']]
    assert study.continuation_gate(fits)['passed']
    fits[3]['top1_teacher_agreement'] = .4  # Circuit seed17 loses by10pp despite positive mean gain.
    result = study.continuation_gate(fits)
    assert not result['passed']
    assert result['checks'][0]['passed'] and not result['checks'][2]['passed']
    fits[3]['top1_teacher_agreement'] = .6
    for row in fits:
        if row['mode'] == 'circuit':
            row['value_mae'] = .3
    assert not study.continuation_gate(fits)['checks'][3]['passed']
    with pytest.raises(ValueError, match='every fixed fit seed'):
        study.continuation_gate(fits[:-1])
