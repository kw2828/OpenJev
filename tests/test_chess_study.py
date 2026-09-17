import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/chess_study.py'
SPEC = importlib.util.spec_from_file_location('chess_study_tested', SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def row(number=0, rating=1000, moves='e2e4 e7e5'):
    return {
        'PuzzleId': f'p{number}', 'FEN': chess.STARTING_FEN, 'Moves': moves,
        'Rating': str(rating), 'GameUrl': f'https://lichess.org/test{number}', 'Themes': 'opening',
    }


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def puzzle_csv(tmp_path):
    path = tmp_path / 'puzzles.csv'
    write_csv(path, [row(i, (1000, 1500, 2000)[i % 3]) for i in range(30)])
    return path


def test_selection_is_source_order_stratified_and_applies_setup(puzzle_csv):
    selected = study.select_puzzles(puzzle_csv)
    assert selected['counts'] == dict.fromkeys(study.BANDS, 8)
    assert [p['puzzle_id'] for p in selected['positions']] == [f'p{i}' for i in range(24)]
    puzzle = selected['positions'][0]
    board = chess.Board(puzzle['solver_fen'])
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert puzzle['gold_first_uci'] == 'e7e5'
    assert puzzle['gold_solution_uci'] == ['e7e5']


def test_bad_solution_and_duplicate_ids_are_excluded(tmp_path):
    path = tmp_path / 'small.csv'
    write_csv(path, [row(0, 1000, 'e2e4 e7e5 e4e6'), row(1, 1000), row(1, 1500),
                     row(2, 1500), row(3, 2000)])
    result = study.select_puzzles(path, per_band=1)
    assert [p['puzzle_id'] for p in result['positions']] == ['p1', 'p2', 'p3']
    assert len(result['encountered_rejections']) == 2


def test_partial_last_row_never_enters_selection(tmp_path):
    path = tmp_path / 'partial.csv'
    write_csv(path, [row(0, 1000), row(1, 1500), row(2, 2000)])
    with path.open('ab') as handle:
        handle.write(b'partial,no,newline')
    result = study.select_puzzles(path, per_band=1)
    assert result['discarded_incomplete_suffix_bytes'] == len(b'partial,no,newline')
    with pytest.raises(ValueError, match='Insufficient'):
        study.select_puzzles(path, per_band=2)


def test_puzzle_policy_gets_only_post_setup_board_and_no_gold():
    puzzle = study.parse_puzzle(row(), 2)

    def policy(board):
        assert type(board) is chess.Board
        assert [move.uci() for move in board.move_stack] == ['e2e4']
        assert board.turn == chess.BLACK
        return {'choice': 'e7e5'}

    result = study.evaluate_puzzle(policy, puzzle)
    assert result['gold_match']
    assert result['error'] is None


def test_invalid_choice_is_retained_as_error_without_fallback():
    puzzle = study.parse_puzzle(row(), 2)
    result = study.evaluate_puzzle(lambda board: {'choice': 'e2e5'}, puzzle)
    assert result['error']['type'] == 'ValueError'
    assert result['response'] is None
    assert result['raw_response'] == {'choice': 'e2e5'}
    assert not result['gold_match']


def test_full_legal_probability_support_required():
    result = study.evaluate_puzzle(
        lambda board: {'choice': 'e7e5', 'probabilities': {'e7e5': 1}}, study.parse_puzzle(row(), 2),
    )
    assert result['error']['type'] == 'ValueError'


def test_immediate_alternative_mate_has_separate_metric():
    # Fool's mate after White's setup move, followed by Qh4#.
    board = chess.Board()
    for move in ('f2f3', 'e7e5'):
        board.push_uci(move)
    source = row(moves='g2g4 d8h4') | {'FEN': board.fen()}
    result = study.evaluate_puzzle(lambda board: {'choice': 'd8h4'}, study.parse_puzzle(source, 2))
    assert result['immediate_checkmate']
    assert result['gold_or_immediate_mate']


def config():
    return {
        'policies': {'material': {'kind': 'greedy'}, 'random': {'kind': 'random', 'seed': 7}},
        'games': [{'id': 'material-random', 'white': 'material', 'black': 'random'}],
        'puzzle_policies': ['material', 'random'], 'max_plies': 2,
    }


@pytest.mark.parametrize('mutate', [
    lambda c: c['games'].append(dict(c['games'][0])),
    lambda c: c['games'][0].update(id='../escape'),
    lambda c: c['games'][0].update(white='unknown'),
    lambda c: c['policies']['material'].update(kind='unknown'),
    lambda c: c.update(clock_seconds=float('nan')),
    lambda c: c.update(max_plies=True),
    lambda c: c.update(puzzle_policies=['material', 'material']),
])
def test_invalid_config_rejected(mutate):
    value = config()
    mutate(value)
    with pytest.raises(ValueError):
        study.normalize_config(value)


def prepared(tmp_path, monkeypatch, puzzle_csv):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    plan_path = tmp_path / 'plan.json'
    study.prepare(config(), puzzle_csv, plan_path)
    return plan_path


def test_preparation_never_instantiates_policy_and_hashes_are_verified(tmp_path, monkeypatch, puzzle_csv):
    monkeypatch.setitem(study.POLICY_FACTORIES, 'greedy', lambda spec: pytest.fail('Inference during prepare'))
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    plan = json.loads(path.read_text())
    study.verify_plan(plan)
    with puzzle_csv.open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Changed artifact_hashes'):
        study.verify_plan(plan)


def test_plan_tamper_rejected(tmp_path, monkeypatch, puzzle_csv):
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    plan = json.loads(path.read_text())
    plan['config']['max_plies'] += 1
    with pytest.raises(ValueError, match='identity'):
        study.verify_plan(plan)


def test_directory_artifact_membership_is_frozen(tmp_path, monkeypatch, puzzle_csv):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    folder = tmp_path / 'assets'
    folder.mkdir()
    (folder / 'a').write_text('first')
    value = config()
    value['policies']['material']['artifacts'] = [str(folder)]
    plan = study.prepare(value, puzzle_csv, tmp_path / 'plan.json')
    (folder / 'b').write_text('added after freeze')
    with pytest.raises(ValueError, match='membership'):
        study.verify_plan(plan)


def test_run_baselines_records_trace_pgn_and_unfinished_not_draw(tmp_path, monkeypatch, puzzle_csv):
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    out = tmp_path / 'execution'
    summary = study.run(path, out)
    assert summary['puzzles']['material']['positions'] == 24
    assert summary['puzzles']['random']['positions'] == 24
    assert summary['scored_game_count'] == 0
    assert summary['unfinished_game_count'] == 1
    assert summary['games'][0]['result'] == '*'
    assert len((out / 'puzzles.jsonl').read_text().splitlines()) == 48
    game = json.loads((out / 'game-material-random.json').read_text())
    assert len(game['moves']) == 2
    assert '[Result "*"]' in (out / 'game-material-random.pgn').read_text()
    receipt = json.loads((out / 'completed.json').read_text())
    assert receipt['puzzle_attempts'] == 48
    assert receipt['outputs']['summary.json'] == study.sha256(out / 'summary.json')
    with pytest.raises(FileExistsError):
        study.run(path, out)


def test_one_game_is_explicit_subset_without_puzzle_calls(tmp_path, monkeypatch, puzzle_csv):
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    result = study.run(path, tmp_path / 'single', game_id='material-random')
    assert not result['entire_prepared_panel']
    assert result['puzzles'] == {}
    assert result['selection'] == {'game_ids': ['material-random'], 'puzzle_policies': []}


def test_puzzles_only_has_no_games(tmp_path, monkeypatch, puzzle_csv):
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    result = study.run(path, tmp_path / 'puzzles', puzzles_only=True)
    assert not result['entire_prepared_panel']
    assert result['games'] == []
    assert result['puzzles']['random']['positions'] == 24


def test_failed_game_not_counted_as_loss_or_draw(tmp_path, monkeypatch, puzzle_csv):
    monkeypatch.setitem(study.POLICY_FACTORIES, 'greedy', lambda spec: lambda board: {'choice': 'illegal'})
    path = prepared(tmp_path, monkeypatch, puzzle_csv)
    result = study.run(path, tmp_path / 'failure', game_id='material-random')
    assert result['failed_game_count'] == 1
    assert result['scored_game_count'] == 0
    assert result['games'][0]['result'] == '*'


def test_warmups_excluded_from_scored_calls_and_load_recorded(tmp_path, monkeypatch, puzzle_csv):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    value = config()
    value['policies']['material']['warmup_calls'] = 2
    study.prepare(value, puzzle_csv, tmp_path / 'plan.json')
    result = study.run(tmp_path / 'plan.json', tmp_path / 'run', game_id='material-random')
    warmups = json.loads((tmp_path / 'run/warmups.json').read_text())
    assert len(warmups) == 2
    assert result['game_policy_latency_all_attempts']['material']['calls'] == 1
    loads = json.loads((tmp_path / 'run/loads.json').read_text())
    assert loads['material']['wall_seconds'] >= 0


def test_stockfish_uses_fixed_nodes_fresh_hash_single_thread(monkeypatch):
    class Engine:
        def __init__(self):
            self.id = {'name': 'Fake Stockfish'}
            self.configurations = []
            self.calls = []
            self.closed = False

        def configure(self, settings):
            self.configurations.append(settings)

        def play(self, board, limit, game):
            self.calls.append((board.fen(), limit.nodes, game))
            return SimpleNamespace(move=chess.Move.from_uci('e2e4'))

        def quit(self):
            self.closed = True

    engine = Engine()
    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda path: engine)
    policy = study.StockfishPolicy({'engine_path': '/fake/stockfish', 'nodes': 1000})
    assert policy(chess.Board())['choice'] == 'e2e4'
    policy(chess.Board())
    assert engine.configurations == [{'Threads': 1, 'Hash': 16}, {'Clear Hash': None}, {'Clear Hash': None}]
    assert [call[1] for call in engine.calls] == [1000, 1000]
    assert engine.calls[0][2] is not engine.calls[1][2]
    policy.close()
    assert engine.closed


@pytest.fixture
def student_checkpoint(tmp_path):
    import torch

    from openjev.research.chess_student import MOVE_VOCAB_SHA256, ChessStudent

    checkpoint = tmp_path/'student.pt'
    model = ChessStudent('gru', seed=17)  # Untrained fixture, never a scored experimental checkpoint.
    torch.save({'mode': 'gru', 'seed': 17, 'state_dict': model.state_dict(),
                'vocabulary_sha256': MOVE_VOCAB_SHA256, 'plan_sha256': 'a'*64}, checkpoint)
    return checkpoint


def student_spec(path):
    return {'kind': 'chess_student', 'checkpoint_path': str(path),
            'expected_training_plan_sha256': 'a'*64, 'mode': 'gru', 'seed': 17}


def student_config(path):
    value = config()
    value['policies']['student'] = student_spec(path)
    value['puzzle_policies'] = ['student']
    value['games'] = [{'id': 'student-material', 'white': 'student', 'black': 'material'}]
    return value


def test_student_factory_loads_exact_declared_checkpoint_and_exposes_complete_legal_probabilities(student_checkpoint):
    policy = study.POLICY_FACTORIES['chess_student'](student_spec(student_checkpoint))
    board = chess.Board()
    response = study.timed_choice(policy, board)
    assert response['error'] is None
    assert set(response['response']['probabilities']) == {move.uci() for move in board.legal_moves}
    assert policy.metadata['mode'] == 'gru' and policy.metadata['seed'] == 17
    assert policy.metadata['checkpoint_sha256'] == study.sha256(student_checkpoint)
    assert policy.metadata['training_plan_sha256'] == 'a'*64
    assert policy.metadata['device'] == 'cpu' and policy.metadata['threads'] == 2


@pytest.mark.parametrize(('changed', 'message'), [
    ({'mode': 'rewired'}, 'mode or seed'), ({'seed': 29}, 'mode or seed'),
    ({'expected_training_plan_sha256': 'b'*64}, 'plan hash'),
])
def test_student_checkpoint_identity_mismatch_fails_without_another_seed(student_checkpoint, changed, message):
    with pytest.raises(ValueError, match=message):
        study.chess_student_factory(student_spec(student_checkpoint) | changed)


@pytest.mark.parametrize('changed', [
    {'seed': None}, {'seed': True}, {'seed': 99}, {'mode': None}, {'mode': 'best'},
    {'expected_training_plan_sha256': 'missing'}, {'expected_training_plan_sha256': 'A'*64},
    {'device': 'mps'}, {'threads': 1}, {'threads': True},
])
def test_student_spec_requires_explicit_frozen_identity_before_preparation(student_checkpoint, changed):
    value = student_config(student_checkpoint)
    value['policies']['student'].update(changed)
    with pytest.raises(ValueError, match='Chess student'):
        study.normalize_config(value)


def test_prepare_binds_student_checkpoint_sources_and_dependencies_without_inference(
    tmp_path, monkeypatch, puzzle_csv, student_checkpoint,
):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    monkeypatch.setitem(study.POLICY_FACTORIES, 'chess_student', lambda spec: pytest.fail('Inference during prepare'))
    plan = study.prepare(student_config(student_checkpoint), puzzle_csv, tmp_path/'student-plan.json')
    source = study.ROOT/'src/openjev/research/chess_student.py'
    assert plan['source_hashes'][str(source)] == study.sha256(source)
    assert str(study.ROOT/'scripts/train_chess_student.py') in plan['source_hashes']
    assert str(study.ROOT/'tests/test_chess_student.py') in plan['source_hashes']
    assert plan['artifact_hashes'][str(student_checkpoint)] == study.sha256(student_checkpoint)
    assert plan['config']['policies']['student']['warmup_calls'] == 2
    assert {'tokenizers', 'huggingface-hub'} <= plan['dependencies']['packages'].keys()
    study.verify_plan(plan)
    original_sha = study.sha256
    monkeypatch.setattr(study, 'sha256', lambda path: '0'*64 if path == source else original_sha(path))
    with pytest.raises(ValueError, match='Changed source_hashes'):
        study.verify_plan(plan)


def test_student_checkpoint_bytes_cannot_change_after_prepare(tmp_path, monkeypatch, puzzle_csv, student_checkpoint):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    plan = study.prepare(student_config(student_checkpoint), puzzle_csv, tmp_path/'student-plan.json')
    with student_checkpoint.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed artifact_hashes'):
        study.verify_plan(plan)


def test_tokenizer_and_hub_versions_are_recorded_as_explicit_dependencies(monkeypatch):
    monkeypatch.setattr(study.importlib.metadata, 'version', lambda name: f'{name}-test-version')
    packages = study.dependencies()['packages']
    assert packages['tokenizers'] == 'tokenizers-test-version'
    assert packages['huggingface-hub'] == 'huggingface-hub-test-version'


def test_untrained_student_fixture_runs_declared_game_with_bound_load_receipt(
    tmp_path, monkeypatch, puzzle_csv, student_checkpoint,
):
    monkeypatch.setattr(study, 'CORE_SOURCES', ('scripts/chess_study.py',))
    plan_path = tmp_path/'student-plan.json'
    study.prepare(student_config(student_checkpoint), puzzle_csv, plan_path)
    result = study.run(plan_path, tmp_path/'student-run', game_id='student-material')
    assert result['unfinished_game_count'] == 1 and result['scored_game_count'] == 0
    loads = json.loads((tmp_path/'student-run/loads.json').read_text())
    assert loads['student']['metadata']['checkpoint_sha256'] == study.sha256(student_checkpoint)
    assert loads['student']['metadata']['training_plan_sha256'] == 'a'*64
    assert loads['student']['metadata']['seed'] == 17
    warmups = json.loads((tmp_path/'student-run/warmups.json').read_text())
    assert len(warmups) == 2 and result['game_policy_latency_all_attempts']['student']['calls'] == 1
