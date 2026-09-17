import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_test_render_chess_replay', ROOT/'scripts/render_chess_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
E4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
E5 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'


def fixture_game():
    return {'initial_fen': START, 'white': 'Synthetic renderer fixture', 'black': {'name': 'Fixture opponent'},
            'caption': 'SYNTHETIC RENDERER TEST. This is not a model result.',
            'initial_clocks': {'white': 300, 'black': 300}, 'clocks': {'white': 0, 'black': 298},
            'result': '0-1', 'termination': 'white_time_forfeit', 'final_fen': E5,
            'moves': [{'fen_before': START, 'fen_after': E4, 'move_uci': 'e2e4', 'san': 'e4', 'latency_ms': 2000,
                       'clocks': {'white': 298, 'black': 300}},
                      {'fen_before': E4, 'fen_after': E5, 'move_uci': 'e7e5', 'san': 'e5', 'latency_ms': 2000,
                       'clocks': {'white': 298, 'black': 298}}],
            'attempts': [{'move_uci': 'g1f3', 'status': 'clock_expired'}]}


def test_frames_keep_every_accepted_move_and_timeout_without_phantom_move():
    game = replay.normalize_game(fixture_game())
    assert [frame['fen'] for frame in game['frames']] == [START, E4, E5, E5]
    assert [frame['ply'] for frame in game['frames']] == [0, 1, 2, 2]
    assert game['frames'][-1]['clocks']['white'] == 0
    assert game['frames'][-1]['final']
    assert game['moves'][1]['label'] == '1... e5'
    assert 'g1f3' not in json.dumps(game)
    assert replay.board_cells(E5)[28] == 'p' and replay.board_cells(E5)[36] == 'P'


@pytest.mark.parametrize('fault', ['chain', 'board', 'clock', 'latency', 'final'])
def test_invalid_trace_fails_before_rendering(fault):
    game = fixture_game()
    if fault == 'chain':
        game['moves'][1]['fen_before'] = START
    elif fault == 'board':
        game['moves'][0]['fen_after'] = '8/8/8/8/8/8/8/7 w - - 0 1'
    elif fault == 'clock':
        game['moves'][0]['clocks']['white'] = -1
    elif fault == 'latency':
        game['moves'][0]['latency_ms'] = float('nan')
    else:
        game['final_fen'] = START
    with pytest.raises(ValueError):
        replay.normalize_game(game)


def test_html_untrusted_caption_cannot_escape_data_script():
    game = fixture_game()
    game['caption'] = '</script><img src=x onerror=alert(1)>'
    page = replay.html_page(replay.normalize_game(game), 'a' * 64)
    assert game['caption'] not in page
    assert '\\u003c/script\\u003e' in page
    assert 'src="http' not in page and 'href="http' not in page


def test_gif_encodes_all_actual_frames_and_preserves_recorded_assets(tmp_path):
    game = tmp_path / 'synthetic-game.json'
    game.write_text(json.dumps(fixture_game()))
    before = game.read_bytes()
    prefix = tmp_path / 'synthetic-replay'
    replay.render(game, prefix)
    assert game.read_bytes() == before
    receipt = json.loads(prefix.with_suffix('.json').read_text())
    assert receipt['accepted_moves'] == 2 and receipt['frames'] == 4
    with Image.open(prefix.with_suffix('.gif')) as gif:
        assert gif.n_frames == 4 and gif.info['loop'] == 0
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info['duration'])
        assert durations == [950, 950, 950, 1900]
    assert prefix.with_suffix('.gif').stat().st_size < 4_000_000
    assert 'SYNTHETIC RENDERER TEST' in prefix.with_suffix('.html').read_text()
    with pytest.raises(FileExistsError):
        replay.render(game, prefix)


def test_legal_validation_rejects_wrong_san_and_handles_special_moves():
    chess = pytest.importorskip('chess')
    game = fixture_game()
    game['moves'][0]['san'] = 'e3'
    with pytest.raises(ValueError, match='mismatched SAN'):
        replay.normalize_game(game)
    board = chess.Board('r3k2r/1P6/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1')
    for uci in ('e5d6', 'e8g8', 'b7b8q'):
        position = copy.deepcopy(fixture_game())
        position['initial_fen'] = board.fen()
        move = chess.Move.from_uci(uci)
        san, before = board.san(move), board.fen()
        board.push(move)
        position['final_fen'] = board.fen()
        position['moves'] = [{'fen_before': before, 'fen_after': board.fen(), 'move_uci': uci, 'san': san,
                              'latency_ms': 10, 'clocks': {'white': 20, 'black': 20}}]
        assert replay.normalize_game(position)['moves'][0]['san'] == san
