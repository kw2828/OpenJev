"""Synthetic mechanics checks, never a chess benchmark or scored holdout."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from openjev.research.chess_spatial import MODES, SpatialChess, piece_targets
from openjev.research.chess_spatial_baselines import simple_baselines, target_categories
from openjev.research.chess_spatial_data import auxiliary_transitions

ROOT = Path(__file__).resolve().parents[1]


def load_study():
    spec = importlib.util.spec_from_file_location('spatial_study_test', ROOT/'scripts/train_chess_spatial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study():
    return load_study()


@pytest.fixture(autouse=True)
def small_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def row(board=None, target=None, name='position-0'):
    board = chess.Board() if board is None else board
    target = min(move.uci() for move in board.legal_moves) if target is None else target
    return {'id': name, 'game_id': 0, 'fen': board.fen(en_passant='fen'),
            'target_uci': target, 'target_value': .125,
            'aux_transitions': auxiliary_transitions(board, 101, 0)}


def test_auxiliary_loss_balances_square_classes_and_ignores_invalid_transition(study):
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.zeros(2, 13, 8, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0, 0] = -2
        logits[1, 0] = -20
    changed = torch.zeros(2, 8, 8, dtype=torch.bool)
    changed[:, 0, 0] = True
    valid = torch.tensor([True, False])
    ce = F.cross_entropy(logits, target, reduction='none')
    expected = (ce[0, 0, 0]+ce[0][~changed[0]].mean())/2
    actual = study.auxiliary_loss(logits, target, changed, valid)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, ce[0].mean())
    actual.backward()
    assert logits.grad[0, :, 0, 0].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    with pytest.raises(ValueError, match='valid auxiliary'):
        study.auxiliary_loss(logits.detach(), target, changed, torch.zeros(2, dtype=torch.bool))


def test_tensors_keep_teacher_out_of_inputs_and_successor_in_original_perspective(study):
    board = chess.Board()
    board.push_uci('e2e4')
    original = row(board, 'e7e5')
    altered = copy.deepcopy(original)
    altered['target_uci'], altered['target_value'] = 'c7c5', -.875
    first, names = study.tensors([original])
    second, other_names = study.tensors([altered])
    assert names == other_names
    for name in ('observations', 'candidates', 'mask', 'current', 'future', 'actions', 'aux_mask'):
        torch.testing.assert_close(first[name], second[name])
    assert first['targets'].item() != second['targets'].item()
    for index, transition in enumerate(original['aux_transitions']):
        successor = chess.Board(transition['next_fen'])
        np.testing.assert_array_equal(first['future'][0, index], piece_targets(successor, board.turn))
        assert not np.array_equal(first['future'][0, index], piece_targets(successor, successor.turn))


def test_reconstruction_changes_only_auxiliary_target(study):
    data, _ = study.tensors([row()])
    model = SpatialChess('recurrent', 17, width=4)
    _, _, hidden = model(data['observations'], data['candidates'], data['mask'])
    predictive = study.auxiliary_forward(model, hidden, data, 'predict')
    reconstruction = study.auxiliary_forward(model, hidden, data, 'reconstruct')
    for index in (0, 2, 3, 4):
        torch.testing.assert_close(predictive[index], reconstruction[index])
    current = data['current'].repeat_interleave(data['actions'].shape[1], 0)
    torch.testing.assert_close(reconstruction[1], current)
    torch.testing.assert_close(predictive[1], data['future'].flatten(0, 1))
    assert torch.count_nonzero(predictive[1] != reconstruction[1]) > 0


def test_zero_aux_weight_preserves_main_gradients_and_aux_branch_gets_zero(study):
    data, _ = study.tensors([row(), row(name='position-1')])
    models = [SpatialChess('recurrent', 17, width=4) for _ in range(2)]
    for index, model in enumerate(models):
        logits, values, hidden = model(data['observations'], data['candidates'], data['mask'])
        loss = F.cross_entropy(logits, data['targets'])+.5*F.mse_loss(values, data['values'])
        if index:
            aux, target, _, changed, valid = study.auxiliary_forward(model, hidden, data, 'recurrent')
            loss = loss+0*study.auxiliary_loss(aux, target, changed, valid)
        loss.backward()
    for (name, plain), (other_name, zero_aux) in zip(models[0].named_parameters(), models[1].named_parameters(), strict=True):
        assert name == other_name
        if name.startswith('aux_head.'):
            assert plain.grad is None
            assert zero_aux.grad is not None and torch.count_nonzero(zero_aux.grad) == 0
        else:
            torch.testing.assert_close(plain.grad, zero_aux.grad, rtol=0, atol=0)


def gate_panel(study):
    return [{'mode': mode, 'seed': seed, 'split': split,
             'top1_teacher_agreement': .40 if mode == 'predict' else .30,
             'future_changed_accuracy': .80}
            for mode in MODES for seed in study.PROTOCOL['seeds'] for split in ('dev', 'shift')]


def test_gate_requires_all_controls_seeds_and_panels(study):
    panel = gate_panel(study)
    assert study.continuation_gate(panel)['passed']
    for invalid in (panel[:-1], panel+[panel[0]], [r for r in panel if r['split'] != 'shift']):
        with pytest.raises(ValueError, match='complete fixed fit panel'):
            study.continuation_gate(invalid)
    panel[0]['mode'] = 'unknown'
    with pytest.raises(ValueError, match='complete fixed fit panel'):
        study.continuation_gate(panel)


def test_gate_enforces_shifted_paired_seed_and_future_prediction(study):
    panel = gate_panel(study)
    target = next(r for r in panel if r['mode'] == 'predict' and r['split'] == 'shift')
    target['top1_teacher_agreement'] = .28
    assert not study.continuation_gate(panel)['passed']
    target['top1_teacher_agreement'] = .40
    for value in panel:
        if value['mode'] == 'predict' and value['split'] == 'shift':
            value['future_changed_accuracy'] = .69
    assert not study.continuation_gate(panel)['passed']


def test_baselines_are_legal_deterministic_and_eval_targets_do_not_choose_moves():
    train = [row(target='e2e4', name=f'train-{i}') for i in range(3)]
    original = row(target='d2d4')
    changed = {**original, 'target_uci': 'e2e4', 'target_value': -.9}
    first, second = simple_baselines(train, [original]), simple_baselines(train, [changed])
    assert first['uniform_random']['metrics']['expected_top1_teacher_agreement'] == .05
    assert first['uniform_random']['predictions'][0]['choice'] is None
    assert first['training_move_frequency']['predictions'][0]['choice'] == 'e2e4'
    assert not first['training_move_frequency']['predictions'][0]['correct']
    assert second['training_move_frequency']['predictions'][0]['correct']
    for name in ('training_move_frequency', 'greedy_material'):
        assert first[name]['predictions'][0]['choice'] == second[name]['predictions'][0]['choice']
        assert chess.Move.from_uci(first[name]['predictions'][0]['choice']) in chess.Board().legal_moves
    assert first['greedy_material']['predictions'][0]['choice'] == 'a2a3'


def test_baseline_ties_ignore_illegal_high_frequency_and_are_sorted_uci():
    board = chess.Board()
    board.push_uci('e2e4')
    train = [row(board, 'e7e5', name='train-black')]
    result = simple_baselines(train, [row()])
    assert result['training_move_frequency']['predictions'][0]['choice'] == 'a2a3'


@pytest.mark.parametrize(('fen', 'target', 'expected'), [
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6', {'capture': True, 'noncapture': False}),
    ('4k3/8/8/8/8/8/8/4K2R w K - 0 1', 'e1g1', {'castling': True, 'capture': False}),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8q', {'promotion': True, 'check': True}),
])
def test_baseline_categories_use_native_special_move_semantics(fen, target, expected):
    board = chess.Board(fen)
    move = chess.Move.from_uci(target)
    assert move in board.legal_moves
    categories = target_categories(board, move)
    for key, value in expected.items():
        assert categories[key] == value
    evidence = simple_baselines([row()], [row(board, target)])
    assert evidence['greedy_material']['predictions'][0]['choice'] in {m.uci() for m in board.legal_moves}
    for name in evidence:
        assert evidence[name]['predictions'][0]['categories'] == categories


def test_baselines_reject_empty_duplicate_and_illegal_panels():
    with pytest.raises(ValueError, match='nonempty'):
        simple_baselines([], [row()])
    with pytest.raises(ValueError, match='Repeated'):
        simple_baselines([row(), row()], [row()])
    with pytest.raises(ValueError, match='legal'):
        simple_baselines([row()], [row(target='e2e5')])


class SyntheticEngine:
    """Zero-valued, sorted-move teacher; no external engine or benchmark data."""

    def __init__(self):
        self.id = {'name': 'Stockfish 19 synthetic test double'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def configure(self, _options):
        pass

    def analyse(self, board, limit, *, info=None, root_moves=None):
        legal = list(board.legal_moves) if root_moves is None else root_moves
        move = min(legal, key=lambda m: m.uci())
        return {'pv': [move], 'score': chess.engine.PovScore(chess.engine.Cp(0), board.turn),
                'nodes': limit.nodes}


@pytest.fixture(scope='module')
def synthetic_run(tmp_path_factory):
    study = load_study()
    protocol = copy.deepcopy(study.PROTOCOL)
    protocol.update({'seeds': [17], 'epochs': 1, 'batch_size': 4,
                     'regret_positions_per_split': 2, 'device': 'cpu'})
    for index, split in enumerate(protocol['splits']):
        split.update({'examples': 8 if split['name'] == 'train' else 4, 'game_cap': 12,
                      'max_plies': 12, 'random_move_probability': 1., 'seed_base': 600100+index*100})
    study.PROTOCOL = protocol
    study.EXCLUSIONS = []
    # Source identity is real; this fixture deliberately excludes unrelated test files.
    study.SOURCES = ['scripts/train_chess_spatial.py', 'src/openjev/research/chess_spatial.py',
                     'src/openjev/research/chess_spatial_data.py',
                     'src/openjev/research/chess_spatial_baselines.py', 'tests/test_chess_spatial_study.py']
    directory = tmp_path_factory.mktemp('spatial-synthetic')
    engine = directory/'synthetic-engine-not-executable'
    engine.write_text('Synthetic unit-test fixture only; never an actual chess engine.\n')
    patch = pytest.MonkeyPatch()
    patch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _path: SyntheticEngine())
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        plan = study.prepare(directory/'plan', engine)
        study.run(plan, directory/'execution')
        result = study.report(plan, directory/'execution', directory/'report')
        yield study, plan, directory, result
    finally:
        patch.undo()
        torch.set_num_threads(previous)


def test_tiny_synthetic_full_training_retains_every_arm_and_final_panel(synthetic_run):
    study, plan, directory, result = synthetic_run
    assert result['status'] == 'completed'
    assert len(result['fits']) == 8
    assert {(r['mode'], r['split']) for r in result['fits']} == {(m, s) for m in MODES for s in ('dev', 'shift')}
    for mode in MODES:
        fit = directory/'execution'/f'{mode}-17'
        receipt = json.loads((fit/'completed.json').read_text())
        assert receipt['updates'] == 2 and receipt['examples_seen'] == 8
        model = SpatialChess.load(fit/'weights.pt', expected_plan_sha256=study.sha(plan))
        assert model.mode == mode
        logs = study.jsonl(fit/'learning.jsonl')
        assert all(r['aux_weight'] == (.25 if mode in ('reconstruct', 'predict') else 0.) for r in logs)
    assert set(result['baselines']) == {'dev', 'shift'}
    assert result['novelty_established'] is False and result['elo_estimate'] is None


def copy_execution(synthetic_run, tmp_path):
    study, plan, directory, _ = synthetic_run
    copied = tmp_path/'execution'
    shutil.copytree(directory/'execution', copied)
    return study, plan, copied


def overwrite_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def rehash_fit(study, execution, fit_name, filename):
    fit = execution/fit_name
    receipt = json.loads((fit/'completed.json').read_text())
    receipt['files'][filename] = study.sha(fit/filename)
    overwrite_json(fit/'completed.json', receipt)
    complete = json.loads((execution/'completed.json').read_text())
    complete['fit_receipts'][fit_name] = study.sha(fit/'completed.json')
    overwrite_json(execution/'completed.json', complete)


def test_report_rejects_changed_file_without_rehash(synthetic_run, tmp_path):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    with (execution/'predict-17/dev.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash'):
        study.report(plan, execution, tmp_path/'report')


@pytest.mark.parametrize('corruption', ['probability', 'future_count', 'nonfinite_value', 'transition_count'])
def test_report_rejects_rehashed_semantically_corrupt_evaluation(synthetic_run, tmp_path, corruption):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    file = execution/'predict-17/dev.json'
    result = json.loads(file.read_text())
    prediction = result['predictions'][0]
    if corruption == 'probability':
        prediction['target_probability'] = .999 if prediction['target_probability'] < .5 else .001
    elif corruption == 'future_count':
        prediction['future_changed_correct'] = prediction['future_changed_count']+1
    elif corruption == 'transition_count':
        prediction['aux_transition_count'] += 1
    else:
        prediction['value'] = 1.5  # Finite but outside the model's bounded output.
    result['metrics'] = study.summarize_predictions(result['predictions'])
    overwrite_json(file, result)
    rehash_fit(study, execution, 'predict-17', 'dev.json')
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path/'report')


def test_report_rejects_rehashed_false_auxiliary_weight(synthetic_run, tmp_path):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    file = execution/'recurrent-17/learning.jsonl'
    logs = study.jsonl(file)
    logs[0]['aux_weight'] = .25
    file.write_text(''.join(json.dumps(r)+'\n' for r in logs))
    rehash_fit(study, execution, 'recurrent-17', 'learning.jsonl')
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path/'report')


def test_report_rejects_missing_fit_even_with_adjusted_receipt_counts(synthetic_run, tmp_path):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    complete = json.loads((execution/'completed.json').read_text())
    del complete['fit_receipts']['cnn-17']
    complete['fits'] -= 1
    complete['total_updates'] -= 2
    overwrite_json(execution/'completed.json', complete)
    with pytest.raises(ValueError, match='fit panel'):
        study.report(plan, execution, tmp_path/'report')
