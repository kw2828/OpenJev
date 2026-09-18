"""Synthetic callbacks only; no trained models, engine calls or scored experiments."""

import io
import json
from collections import Counter

import chess
import chess.pgn
import pytest
import torch

from openjev.research import chess_continuation_arena as arena

PLAN = 'a' * 64


def response(board, spec, choice=None):
    legal = sorted(move.uci() for move in board.legal_moves)
    choice = legal[0] if choice is None else choice
    side = 'white' if board.turn else 'black'
    return {'choice': choice, 'probabilities': {uci: float(uci == choice) for uci in legal},
            'width': 32, 'seed': spec['seed'], 'depth': 4, 'mode': 'recurrent',
            'recurrence': 'residual', 'objective': spec[f'{side}_objective']}


def timer(seconds=.1):
    current = 0.

    def tick():
        nonlocal current
        current += seconds
        return current
    return tick


def test_fixed_schedule_and_truthful_width_identities():
    specs = arena.schedule()
    assert len(specs) == 192 and len({s['id'] for s in specs}) == 192
    assert Counter(s['opponent'] for s in specs) == {'policy': 96, 'best_value': 96}
    assert Counter(s['seed'] for s in specs) == {193: 64, 211: 64, 227: 64}
    for opening in arena.OPENINGS:
        for seed in arena.SEEDS:
            for opponent in arena.OPPONENTS:
                pair = [s for s in specs if s['opening_id'] == opening['id']
                        and s['seed'] == seed and s['opponent'] == opponent]
                assert len(pair) == 2
                assert pair[0]['white'] == pair[1]['black']
                assert pair[0]['black'] == pair[1]['white']
                assert all(s['white_width'] == s['black_width'] == 32 for s in pair)
    rules = arena.protocol()
    assert rules['max_played_plies'] == 240 and rules['clock_seconds'] == 300
    assert rules['increment'] == 0 and rules['depth'] == 4 and rules['claim_draw'] is False
    assert rules['openings'] == arena.native.protocol()['openings']


def test_native_kernel_parity_full_history_and_cap(monkeypatch):
    monkeypatch.setattr(arena.native, 'MAX_PLAYED_PLIES', 2)
    spec = arena.schedule()[0]
    histories = []

    def policy(board):
        histories.append([move.uci() for move in board.move_stack])
        value = response(board, spec)
        board.clear()
        return value

    expected = arena.native.play(spec, policy, policy, timer=timer())
    game = arena.play(spec, policy, policy, timer=timer())
    assert all(game[key] == value for key, value in expected.items())
    assert histories[0] == spec['opening_moves']
    assert histories[1][:6] == spec['opening_moves'] and len(histories[1]) == 7
    assert game['status'] == 'unfinished' and game['result'] == '*'
    assert game['played_plies'] == 2 and game['wall_seconds'] == pytest.approx(.2)
    assert game['clocks'] == pytest.approx({'white': 299.9, 'black': 299.9})
    arena.validate_game(game, spec)
    pgn = chess.pgn.read_game(io.StringIO(arena.to_pgn(game)))
    assert len(list(pgn.mainline_moves())) == 8 and pgn.end().board().fen() == game['final_fen']


def test_automatic_repetition_has_complete_opening_history():
    spec = arena.schedule()[0]

    def repeat(board):
        choice = ('f3g1', 'g8f6', 'g1f3', 'f6g8')[(len(board.move_stack) - 6) % 4]
        return response(board, spec, choice)

    game = arena.play(spec, repeat, repeat, timer=timer())
    assert (game['status'], game['result'], game['termination']) == (
        'completed', '1/2-1/2', 'fivefold_repetition')
    assert game['played_plies'] == 16
    arena.validate_game(game, spec)


@pytest.mark.parametrize('taken', [True, False])
def test_mate_diagnostic_records_taken_or_missed_without_policy_feedback(monkeypatch, taken):
    monkeypatch.setattr(arena.native, 'MAX_PLAYED_PLIES', 1)
    spec = arena.schedule()[0]
    spec['opening_moves'] = ['e2e4', 'e7e5', 'd1h5', 'b8c6', 'f1c4', 'g8f6']
    choice = 'h5f7' if taken else 'h5h3'
    calls = []

    def policy(board):
        calls.append(board.fen())
        return response(board, spec, choice)

    game = arena.play(spec, policy, policy, timer=timer())
    assert len(calls) == 1 and game['moves'][-1]['move_uci'] == choice
    diagnostic = game['mate_diagnostic']['by_player'][spec['white']]
    assert diagnostic == {'accepted_turns': 1, 'mate_opportunities': 1,
                          'mate_taken': int(taken), 'mate_missed': int(not taken)}
    assert game['status'] == ('completed' if taken else 'unfinished')
    assert game['wall_seconds'] == pytest.approx(.1)
    arena.validate_game(game, spec)
    game['mate_diagnostic']['opportunities'][0]['taken'] = not taken
    with pytest.raises(ValueError, match='diagnostic'):
        arena.validate_game(game, spec)


def test_timeout_proposal_discarded_not_fallback():
    spec = arena.schedule()[0]
    policy = lambda board: response(board, spec)
    game = arena.play(spec, policy, policy, timer=timer(301))
    assert game['termination'] == 'timeout' and game['result'] == '0-1'
    assert game['played_plies'] == 0 and len(game['attempts']) == 1
    assert sum(v['accepted_turns'] for v in game['mate_diagnostic']['by_player'].values()) == 0
    arena.validate_game(game, spec)
    game['attempts'][0]['policy']['objective'] = 'policy'
    with pytest.raises(ValueError, match='objective'):
        arena.validate_game(game, spec)


@pytest.mark.parametrize('fault', ['exception', 'objective', 'choice', 'menu', 'seed', 'argmax'])
def test_invalid_callback_is_recorded_once_without_retry(fault):
    spec = arena.schedule()[0]
    calls = []

    def bad(board):
        calls.append(1)
        if fault == 'exception':
            raise RuntimeError('Synthetic callback failure')
        value = response(board, spec)
        if fault == 'objective':
            value['objective'] = 'policy'
        elif fault == 'choice':
            value['choice'] = 'a1a8'
        elif fault == 'menu':
            value['probabilities'].pop(next(iter(value['probabilities'])))
        elif fault == 'seed':
            value['seed'] = 17
        else:
            value['choice'] = max(value['probabilities'])
        return value

    game = arena.play(spec, bad, bad, timer=timer())
    assert game['status'] == 'failed' and game['result'] == '*'
    assert game['played_plies'] == 0 and len(calls) == 1
    assert game['attempts'][0]['disposition'] in ('policy_error', 'invalid_response', 'invalid_choice')
    arena.validate_game(game, spec)


def invented_results(wins=58, failures=0):
    games = []
    for i, spec in enumerate(arena.schedule()):
        within = i % 96
        game = {**spec, 'game_id': spec['id'], 'attempts': [], 'played_plies': 0,
                'wall_seconds': 0., 'mate_diagnostic': {'by_player': {}}}
        if within < wins:
            game.update(status='completed', termination='checkmate',
                        result='1-0' if spec['white_objective'] == 'continuation' else '0-1')
        else:
            game.update(status='failed' if within < wins + failures else 'unfinished',
                        termination='max_plies', result='*')
        games.append(game)
    return games


def test_all_game_bounds_and_bootstrap_never_turn_caps_into_draws():
    games = invented_results()
    summary = arena.summarize(games)
    for panel in summary['by_opponent'].values():
        assert panel['continuation']['score_lower_bound'] == 58 / 96
        assert panel['continuation']['score_upper_bound'] == 1.
        assert panel['continuation']['draws'] == 0
        assert panel['opening_cluster_bootstrap']['upper_bound_interval'] == [1., 1.]
    assert summary['gate_passed'] and summary['elo_estimate'] is None
    assert summary == arena.summarize(games)
    assert not arena.summarize(invented_results(wins=57))['gate_passed']
    assert not arena.summarize(invented_results(failures=1))['gate_passed']
    assert summary['status_counts'] == {'completed': 116, 'unfinished': 76, 'failed': 0}


@pytest.mark.parametrize('fault', ['missing', 'order', 'identity', 'status', 'result', 'protocol'])
def test_summary_rejects_incomplete_or_misidentified_games(fault):
    games, rules = invented_results(), arena.protocol()
    if fault == 'missing':
        games.pop()
    elif fault == 'order':
        games[0], games[1] = games[1], games[0]
    elif fault == 'identity':
        games[0]['white_objective'] = 'policy'
    elif fault == 'status':
        games[0]['status'] = 'unknown'
    elif fault == 'result':
        games[-1]['result'] = '1/2-1/2'
    else:
        rules['gate_lower_score'] = .59
    with pytest.raises(ValueError):
        arena.summarize(games, rules)


class SyntheticModel(torch.nn.Module):
    def __init__(self, objective, seed):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.tensor(0.))
        self.objective, self.seed = objective, seed
        self.width, self.depth, self.recurrence = 32, 4, 'residual'
        self.eval()

    def choose(self, board, depth=None):
        assert depth == 4 and len(board.move_stack) >= 6 and torch.get_num_threads() == 2
        value = response(board, {'seed': self.seed, 'white_objective': self.objective,
                                 'black_objective': self.objective})
        del value['objective']  # Match inherited actor API; wrapper adds truthful identity.
        return value


def models():
    return {arena.model_name(o, s): SyntheticModel(o, s) for o in arena.OBJECTIVES for s in arena.SEEDS}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, 'ContinuationChess', SyntheticModel)
    monkeypatch.setattr(arena.native, 'MAX_PLAYED_PLIES', 2)
    previous = torch.get_num_threads()
    path = tmp_path / 'arena'
    arena.run(models(), path, PLAN)
    yield path
    torch.set_num_threads(previous)


def test_saved_roundtrip_exact_schedule_models_pgn_and_no_overwrite(saved):
    summary = arena.report(saved, PLAN)
    assert summary['games'] == 192 and summary['status_counts']['unfinished'] == 192
    assert summary['policy_calls'] == summary['played_plies'] == 384
    assert sum(v['accepted_turns'] for v in summary['mate_diagnostic_by_player'].values()) == 384
    assert not summary['gate_passed']
    assert arena.audit(saved, PLAN) == summary
    with pytest.raises(FileExistsError):
        arena.run(models(), saved, PLAN)
    with pytest.raises(ValueError):
        arena.report(saved, 'b' * 64)


@pytest.mark.parametrize('fault', ['objective', 'seed', 'train_mode', 'nan', 'missing'])
def test_model_bindings_reject_invalid_stored_identities(monkeypatch, fault):
    monkeypatch.setattr(arena, 'ContinuationChess', SyntheticModel)
    values = models()
    model = values['continuation-193']
    if fault == 'objective':
        model.objective = 'policy'
    elif fault == 'seed':
        model.seed = 17
    elif fault == 'train_mode':
        model.train()
    elif fault == 'nan':
        model.placeholder.data.fill_(float('nan'))
    else:
        values.pop('policy-193')
    with pytest.raises(ValueError):
        arena._model_bindings(values)


def test_rehashed_identity_corruption_is_rejected(saved):
    path = saved / 'game-001.json'
    game = json.loads(path.read_text())
    game['attempts'][0]['policy']['objective'] = 'policy'
    game['moves'][6]['policy']['objective'] = 'policy'
    path.write_text(json.dumps(game))
    receipt_path = saved / 'completed.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['files'][path.name] = arena._hash(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='objective'):
        arena.report(saved, PLAN)
