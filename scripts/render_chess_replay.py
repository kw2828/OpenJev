"""Render a recorded chess game as a self-contained HTML replay, GIF and PNG.

Only accepted moves from ``moves`` are rendered. ``attempts`` never supplies a
board position. Pillow is required; python-chess, when installed, also verifies
every recorded transition and SAN. No engine, model or network call is made.
"""

import argparse
import hashlib
import html
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GLYPHS = dict(zip('KQRBNPkqrbnp', '♚♛♜♝♞♟♚♛♜♝♞♟', strict=True))
PIECE_NAMES = dict(zip('kqrbnp', ('king', 'queen', 'rook', 'bishop', 'knight', 'pawn'), strict=True))
RESULTS = {'1-0', '0-1', '1/2-1/2', '*'}
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/System/Library/Fonts/Apple Symbols.ttf',
]
FRAME_MS = 950
MAX_GIF_BYTES = 4_000_000


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))


def board_cells(fen):
    """Return 64 piece codes, ordered a8..h1, without assuming a starting layout."""
    if not isinstance(fen, str):
        raise TypeError('FEN must be a string')
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ('w', 'b'):
        raise ValueError('Expected a complete six-field FEN')
    ranks = fields[0].split('/')
    if len(ranks) != 8 or not fields[4].isdigit() or not fields[5].isdigit() or int(fields[5]) < 1:
        raise ValueError('Invalid FEN board or counters')
    cells = []
    for rank in ranks:
        row = []
        for piece in rank:
            if piece in '12345678':
                row.extend([''] * int(piece))
            elif piece in GLYPHS:
                row.append(piece)
            else:
                raise ValueError('Invalid FEN piece')
        if len(row) != 8:
            raise ValueError('Every FEN rank must describe eight squares')
        cells.extend(row)
    return cells


def clocks(value):
    if value is None:
        return {'white': None, 'black': None}
    if not isinstance(value, dict) or not {'white', 'black'} <= value.keys():
        raise ValueError('Clocks must contain white and black seconds')
    result = {}
    for side in ('white', 'black'):
        seconds = value[side]
        if seconds is not None and (type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('Clock seconds must be nonnegative and finite, or null')
        result[side] = seconds
    return result


def clock_text(seconds):
    if seconds is None:
        return '--:--'
    tenths = round(seconds * 10)
    return f'{tenths // 600:02d}:{tenths // 10 % 60:02d}.{tenths % 10}'


def player_name(value, side):
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ('name', 'label', 'model', 'id'):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
    raise ValueError(f'Expected a player name or named object for {side}')


def normalize_game(game, caption=None):
    if not isinstance(game, dict):
        raise TypeError('Expected one game trace object')
    initial = game['initial_fen']
    board_cells(initial)
    result = game['result']
    if result not in RESULTS:
        raise ValueError('Unknown chess result')
    termination = game['termination']
    if not isinstance(termination, str) or not termination:
        raise ValueError('Expected a recorded termination reason')
    moves = game['moves']
    if not isinstance(moves, list):
        raise TypeError('Accepted moves must be a list')
    try:
        import chess
    except ImportError:
        chess = None
    legal_board = chess.Board(initial) if chess else None
    if legal_board is not None and not legal_board.is_valid():
        raise ValueError('Initial chess position is invalid')
    normalized = {'white': player_name(game['white'], 'white'), 'black': player_name(game['black'], 'black'),
                  'result': result, 'termination': termination, 'moves': [],
                  'caption': caption or game.get('caption') or
                  'Recorded OpenJev game. Player labels identify the move selectors. This replay makes no strength claim.',
                  'validation': 'legal moves, SAN and FEN verified with python-chess' if chess else
                  'FEN structure and trace continuity checked; python-chess unavailable'}
    if not isinstance(normalized['caption'], str):
        raise TypeError('Caption must be plain text')
    current_fen = initial
    initial_clocks = clocks(game.get('initial_clocks'))
    frames = [{'fen': initial, 'clocks': initial_clocks, 'ply': 0, 'label': 'Starting position',
               'last_move': None, 'latency_ms': None, 'final': False}]
    for index, move in enumerate(moves, 1):
        if move['fen_before'] != current_fen:
            raise ValueError(f'Broken FEN chain at accepted move {index}')
        board_cells(move['fen_after'])
        uci, san = move['move_uci'], move['san']
        if (not isinstance(uci, str) or len(uci) not in (4, 5)
                or any(uci[i] not in 'abcdefgh' for i in (0, 2))
                or any(uci[i] not in '12345678' for i in (1, 3))
                or (len(uci) == 5 and uci[4] not in 'qrbn') or not isinstance(san, str) or not san):
            raise ValueError('Invalid recorded UCI or SAN')
        latency = move.get('latency_ms')
        if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise ValueError('Move latency must be finite and nonnegative, or null')
        if legal_board is not None:
            parsed = chess.Move.from_uci(uci)
            if parsed not in legal_board.legal_moves or legal_board.san(parsed) != san:
                raise ValueError(f'Illegal move or mismatched SAN at ply {index}')
            legal_board.push(parsed)
            if legal_board.fen() != chess.Board(move['fen_after']).fen():
                raise ValueError(f'Move does not produce recorded FEN at ply {index}')
        before_fields = current_fen.split()
        label = f'{before_fields[5]}{"." if before_fields[1] == "w" else "..."} {san}'
        entry = {'fen_before': current_fen, 'fen_after': move['fen_after'], 'move_uci': uci, 'san': san,
                 'label': label, 'latency_ms': latency, 'clocks': clocks(move.get('clocks'))}
        normalized['moves'].append(entry)
        frames.append({'fen': move['fen_after'], 'clocks': entry['clocks'], 'ply': index, 'label': label,
                       'last_move': uci, 'latency_ms': latency, 'final': False})
        current_fen = move['fen_after']
    final_fen = game.get('final_fen', game.get('fen', current_fen))
    if final_fen != current_fen:
        raise ValueError('Final position differs from the last accepted move')
    # A flag fall may change the clocks without creating another legal move.
    final_clocks = clocks(game.get('clocks', game.get('final_clocks', frames[-1]['clocks'])))
    frames.append({**frames[-1], 'clocks': final_clocks, 'label': f'Final: {termination}', 'final': True})
    normalized['frames'] = frames
    return normalized


def font_path():
    for filename in FONT_CANDIDATES:
        if Path(filename).is_file():
            return filename
    try:
        # Pillow can locate a system DejaVu installation not in the paths above.
        loaded = ImageFont.truetype('DejaVuSans.ttf', 24)
        return loaded.path
    except OSError as error:
        raise RuntimeError('Install a local DejaVu Sans or Arial Unicode font to render chess pieces') from error


def shortened(draw, text, font, max_width):
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + '…', font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + '…'


def wrapped(draw, text, font, width, max_lines):
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines[max_lines - 1] = shortened(draw, ' '.join(lines[max_lines - 1:]), font, width)
    return lines[:max_lines]


def draw_frame(game, frame, font_file, width=640):
    """Draw real FEN pieces at a fixed layout, then resize mechanically if needed."""
    image = Image.new('RGB', (640, 1030), '#111b26')
    draw = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(font_file, size) for size in (12, 13, 14, 16, 18, 22, 26, 30, 58)}
    draw.text((28, 17), 'OPENJEV / RECORDED CHESS', font=fonts[14], fill='#7cd7c1')
    draw.text((28, 44), 'Local model at the board', font=fonts[26], fill='#f4f6f8')
    for index, line in enumerate(wrapped(draw, game['caption'], fonts[13], 584, 3)):
        draw.text((28, 84 + 17 * index), line, font=fonts[13], fill='#afbecd')
    active = frame['fen'].split()[1]
    for side, y, color_code in [('black', 145, 'b'), ('white', 811, 'w')]:
        playing = not frame['final'] and active == color_code
        draw.rounded_rectangle((28, y, 612, y + 61), radius=12, fill='#1e2b39',
                               outline='#7cd7c1' if playing else '#344251', width=2 if playing else 1)
        draw.text((43, y + 8), side.upper() + (' / TO MOVE' if playing else ''), font=fonts[12], fill='#94a9bd')
        name = shortened(draw, game[side], fonts[18], 392)
        draw.text((43, y + 27), name, font=fonts[18], fill='#f4f6f8')
        draw.text((593, y + 31), clock_text(frame['clocks'][side]), font=fonts[26], fill='#f4f6f8', anchor='rm')
    left, top, square = 32, 221, 72
    highlight = {frame['last_move'][:2], frame['last_move'][2:4]} if frame['last_move'] else set()
    for index, piece in enumerate(board_cells(frame['fen'])):
        row, col = divmod(index, 8)
        name = f'{chr(97 + col)}{8 - row}'
        dark = (row + col) % 2 == 1
        fill = '#647f90' if dark else '#e9e4d7'
        if name in highlight:
            fill = '#a9b478' if dark else '#d5d49b'
        x, y = left + col * square, top + row * square
        draw.rectangle((x, y, x + square - 1, y + square - 1), fill=fill)
        if piece:
            white = piece.isupper()
            draw.text((x + square / 2, y + square / 2 + 1), GLYPHS[piece], font=fonts[58], anchor='mm',
                      fill='#faf8ef' if white else '#202c39', stroke_width=1 if white else 0,
                      stroke_fill='#263547')
        if col == 0:
            draw.text((x + 4, y + 2), str(8 - row), font=fonts[12], fill='#344956' if not dark else '#f1eee4')
        if row == 7:
            draw.text((x + square - 12, y + square - 17), chr(97 + col), font=fonts[12],
                      fill='#344956' if not dark else '#f1eee4')
    label = shortened(draw, frame['label'], fonts[22], 584)
    draw.text((28, 896), label, font=fonts[22], fill='#7cd7c1' if not frame['final'] else '#e9c78c')
    total = len(game['moves'])
    detail = f'Played moves: {frame["ply"]}/{total}'
    if frame['latency_ms'] is not None and not frame['final']:
        detail += f'  /  Decision: {frame["latency_ms"] / 1000:.2f}s'
    draw.text((28, 928), detail, font=fonts[14], fill='#afbecd')
    status = f'Recorded result: {game["result"]}  /  {game["termination"].replace("_", " ")}'
    draw.text((28, 955), shortened(draw, status, fonts[14], 584), font=fonts[14], fill='#afbecd')
    draw.text((28, 991), 'Actual positions. Recorded clocks. Replay timing is accelerated.', font=fonts[12], fill='#899bab')
    if width != 640:
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
    return image


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenJev · recorded chess</title><style>
:root{color-scheme:dark;--bg:#101a25;--card:#1b2a38;--line:#334659;--text:#f3f5f7;--muted:#a4b5c6;--mint:#7bd7bf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1140px;margin:auto;padding:35px 24px 28px}.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--mint);font-weight:700}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;margin:10px 0 14px;letter-spacing:-.025em}.caption{color:var(--muted);max-width:900px;margin:0 0 27px}
.layout{display:grid;grid-template-columns:minmax(0,650px) minmax(235px,1fr);gap:28px;align-items:start}.player{display:flex;gap:16px;align-items:center;justify-content:space-between;padding:12px 17px;background:var(--card);border:1px solid var(--line);border-radius:12px}.player.active{border-color:var(--mint);box-shadow:inset 3px 0 var(--mint)}
.side{font-size:11px;letter-spacing:.09em;color:var(--muted)}.name{font-weight:600;overflow-wrap:anywhere}.clock{font:600 clamp(22px,3vw,30px)/1.1 ui-monospace,Menlo,monospace;white-space:nowrap;font-variant-numeric:tabular-nums}
.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;margin:14px 0;border-radius:5px;overflow:hidden;box-shadow:0 8px 30px #0003}.square{position:relative;display:flex;align-items:center;justify-content:center;aspect-ratio:1;background:#e9e4d7}.square.dark{background:#647f90}.square.last{background:#d5d49b}.square.dark.last{background:#a9b478}
.piece{font-family:'DejaVu Sans','Arial Unicode MS','Apple Symbols','Segoe UI Symbol',serif;font-size:clamp(27px,6.5vw,65px);line-height:1;color:#202c39}.piece.white{color:#faf8ef;-webkit-text-stroke:1px #263547;text-shadow:0 1px 1px #24324155}.coord{position:absolute;font:600 11px/1 system-ui;color:#344956}.dark .coord{color:#f1eee4}.rank{left:4px;top:4px}.file{right:4px;bottom:4px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}.panel+.panel{margin-top:15px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.result{font-size:30px;font-weight:650;margin:4px 0}.reason{color:var(--muted);overflow-wrap:anywhere}.last-move{font-size:22px;font-weight:650;margin:5px 0;color:var(--mint)}.detail{font-size:13px;color:var(--muted)}
.controls{display:flex;gap:9px;margin:19px 0 12px}button{background:#283c4d;border:1px solid #466074;border-radius:8px;color:var(--text);font:inherit;cursor:pointer;min-height:42px;padding:6px 15px}button:hover{background:#365065}button:focus-visible,input:focus-visible{outline:3px solid var(--mint);outline-offset:3px}button:disabled{opacity:.4;cursor:default}#play{background:var(--mint);color:#11272b;border-color:var(--mint);font-weight:650;min-width:94px}.scrub{width:100%;accent-color:var(--mint);margin:4px 0 7px}.progress{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}.move-list{display:grid;grid-template-columns:1fr 1fr;gap:6px;max-height:235px;overflow-y:auto;margin-top:12px}.move-list button{text-align:left;min-height:36px;font-size:13px;padding:5px 8px;border-color:transparent;background:#233544}.move-list button[aria-current=step]{border-color:var(--mint);color:var(--mint)}
footer{margin-top:23px;color:#8e9fae;font-size:12px}.note{margin:12px 0 0;color:var(--muted);font-size:12px}code{overflow-wrap:anywhere}noscript{display:block;padding:24px;background:#442f21}@media(max-width:760px){main{padding:24px 15px}.layout{grid-template-columns:1fr;gap:19px}.sidebar{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel+.panel{margin:0}.panel{padding:15px}.piece{font-size:clamp(29px,9vw,65px)}}@media(max-width:420px){.sidebar{grid-template-columns:1fr}.coord{font-size:9px}.player{padding:10px 12px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><main><div class="eyebrow">OPENJEV / RECORDED CHESS</div><h1>Local model at the board</h1><p class="caption" id="caption"></p>
<noscript>This offline replay needs JavaScript for navigation. The accompanying PNG shows the final position and the GIF shows all recorded moves.</noscript>
<div class="layout"><section aria-label="Chess replay"><div class="player" id="black-player"><div><div class="side">BLACK</div><div class="name" id="black-name"></div></div><div class="clock" id="black-clock" aria-label="Recorded black clock"></div></div>
<div id="board" class="board" role="grid" aria-label="Chess position, white at bottom"></div>
<div class="player" id="white-player"><div><div class="side">WHITE</div><div class="name" id="white-name"></div></div><div class="clock" id="white-clock" aria-label="Recorded white clock"></div></div>
<div class="controls"><button id="prev" aria-label="Previous recorded position">← Prev</button><button id="play" aria-label="Play replay">Play</button><button id="next" aria-label="Next recorded position">Next →</button></div>
<label class="label" for="scrub">Recorded position</label><input id="scrub" class="scrub" type="range" min="0" value="0" step="1"><div class="progress"><output id="position" for="scrub"></output><span>Arrow keys to step</span></div></section>
<aside class="sidebar"><section class="panel"><div class="label">Recorded result</div><div id="result" class="result"></div><div id="termination" class="reason"></div><hr style="border:0;border-top:1px solid var(--line);margin:19px 0"><div class="label">Current position</div><div id="last-move" class="last-move" aria-live="polite"></div><div id="latency" class="detail"></div><p class="note">Clocks are recorded snapshots. Replay advances every 0.95 seconds, independently of decision time.</p></section>
<section class="panel"><div class="label">Played moves</div><div id="moves" class="move-list" aria-label="Jump to a played move"></div><p class="note">Only accepted moves are shown. An expired clock can end the game without another move.</p></section></aside></div>
<footer><span id="validation"></span>. Fully offline replay; no model or chess engine runs in this page.<br>Source trace SHA-256: <code>__SHA__</code></footer>
</main><script id="replay-data" type="application/json">__DATA__</script><script>
'use strict';
const game=JSON.parse(document.getElementById('replay-data').textContent);
const byId=id=>document.getElementById(id), glyph={K:'♚',Q:'♛',R:'♜',B:'♝',N:'♞',P:'♟',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const names={k:'king',q:'queen',r:'rook',b:'bishop',n:'knight',p:'pawn'};
let index=0,timer=null;
function clock(seconds){if(seconds===null)return '--:--';const n=Math.round(seconds*10);return String(Math.floor(n/600)).padStart(2,'0')+':'+String(Math.floor(n/10)%60).padStart(2,'0')+'.'+n%10;}
function stop(){if(timer!==null)clearInterval(timer);timer=null;byId('play').textContent='Play';byId('play').setAttribute('aria-label','Play replay');}
function show(next){index=Math.max(0,Math.min(game.frames.length-1,next));const f=game.frames[index];let cells=[];
 for(const c of f.fen.split(' ')[0]){if(c==='/')continue;if(/[1-8]/.test(c))cells.push(...Array(Number(c)).fill(''));else cells.push(c);}
 const last=f.last_move?[f.last_move.slice(0,2),f.last_move.slice(2,4)]:[];byId('board').replaceChildren();
 cells.forEach((p,i)=>{const row=Math.floor(i/8),col=i%8,square=String.fromCharCode(97+col)+(8-row),el=document.createElement('div');el.className='square'+((row+col)%2?' dark':'')+(last.includes(square)?' last':'');el.setAttribute('role','gridcell');el.setAttribute('aria-label',square+(p?': '+(p===p.toUpperCase()?'white ':'black ')+names[p.toLowerCase()]:': empty'));
 if(p){const piece=document.createElement('span');piece.className='piece'+(p===p.toUpperCase()?' white':'');piece.textContent=glyph[p];piece.setAttribute('aria-hidden','true');el.append(piece);}
 if(col===0){const rank=document.createElement('span');rank.className='coord rank';rank.textContent=8-row;rank.setAttribute('aria-hidden','true');el.append(rank);}if(row===7){const file=document.createElement('span');file.className='coord file';file.textContent=String.fromCharCode(97+col);file.setAttribute('aria-hidden','true');el.append(file);}byId('board').append(el);});
 const turn=f.fen.split(' ')[1];for(const side of ['white','black']){byId(side+'-clock').textContent=clock(f.clocks[side]);byId(side+'-player').classList.toggle('active',!f.final&&turn===(side==='white'?'w':'b'));}
 byId('last-move').textContent=f.label;byId('latency').textContent=f.latency_ms===null||f.final?'Played moves: '+f.ply+'/'+game.moves.length:'Decision: '+(f.latency_ms/1000).toFixed(2)+'s · Played moves: '+f.ply+'/'+game.moves.length;
 byId('position').textContent=f.final?'Final recorded position':'Position '+index+' / '+(game.frames.length-1);byId('scrub').value=index;byId('prev').disabled=index===0;byId('next').disabled=index===game.frames.length-1;
 Array.from(byId('moves').children).forEach((button,i)=>{if(!f.final&&i+1===index)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});if(index===game.frames.length-1)stop();}
function play(){if(timer!==null){stop();return;}if(index===game.frames.length-1)show(0);byId('play').textContent='Pause';byId('play').setAttribute('aria-label','Pause replay');timer=setInterval(()=>show(index+1),950);}
byId('caption').textContent=game.caption;byId('white-name').textContent=game.white;byId('black-name').textContent=game.black;byId('result').textContent=game.result;byId('termination').textContent=game.termination.replaceAll('_',' ');byId('validation').textContent=game.validation;
byId('scrub').max=game.frames.length-1;byId('scrub').addEventListener('input',event=>{stop();show(Number(event.target.value));});byId('prev').onclick=()=>{stop();show(index-1);};byId('next').onclick=()=>{stop();show(index+1);};byId('play').onclick=play;
game.moves.forEach((move,i)=>{const button=document.createElement('button');button.textContent=move.label;button.setAttribute('aria-label','Show position after '+move.label);button.onclick=()=>{stop();show(i+1);};byId('moves').append(button);});
document.addEventListener('keydown',event=>{if(event.target.matches('input,button'))return;if(['ArrowLeft','ArrowRight','Home','End',' '].includes(event.key)){event.preventDefault();if(event.key===' ')play();else{stop();show(event.key==='Home'?0:event.key==='End'?game.frames.length-1:index+(event.key==='ArrowLeft'?-1:1));}}});show(0);
</script></body></html>'''


def html_page(game, source_sha):
    # Do not allow a trace caption or player label to escape its JSON script.
    data = json.dumps(game, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return HTML.replace('__SHA__', html.escape(source_sha)).replace('__DATA__', data)


def render(game_path, out, caption=None):
    game = normalize_game(read_json(game_path), caption)
    paths = {suffix: Path(str(out) + '.' + suffix) for suffix in ('html', 'gif', 'png', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Choose a new output prefix; recorded replays are never overwritten')
    font_file = font_path()
    original_frames = [draw_frame(game, frame, font_file) for frame in game['frames']]
    durations = [FRAME_MS] * (len(original_frames) - 1) + [1900]
    gif_bytes, settings = None, None
    for width, colors in [(640, 96), (560, 64), (480, 48), (400, 32)]:
        size = (width, round(original_frames[0].height * width / original_frames[0].width))
        frames = [frame.resize(size, Image.Resampling.LANCZOS).quantize(colors=colors,
                  method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for frame in original_frames]
        stream = io.BytesIO()
        frames[0].save(stream, format='GIF', save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=1)
        if stream.tell() <= MAX_GIF_BYTES:
            gif_bytes, settings = stream.getvalue(), {'width': width, 'height': size[1], 'colors': colors}
            break
    if gif_bytes is None:
        raise ValueError('Full-move GIF exceeds 4 MB; do not silently drop moves')
    with Image.open(io.BytesIO(gif_bytes)) as decoded:
        if decoded.n_frames != len(original_frames):
            raise ValueError('GIF encoder collapsed a recorded frame')
        actual_durations = []
        for index in range(decoded.n_frames):
            decoded.seek(index)
            actual_durations.append(decoded.info['duration'])
        if actual_durations != durations or decoded.info.get('loop') != 0:
            raise ValueError('GIF timing or loop changed')
    png = io.BytesIO()
    original_frames[-1].save(png, format='PNG', optimize=True)
    source_sha = hashlib.sha256(game_path.read_bytes()).hexdigest()
    blobs = {'gif': gif_bytes, 'png': png.getvalue(), 'html': html_page(game, source_sha).encode()}
    receipt = {'source_sha256': source_sha, 'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'font_file': str(font_file), 'font_sha256': hashlib.sha256(Path(font_file).read_bytes()).hexdigest(),
               'accepted_moves': len(game['moves']), 'frames': len(original_frames), 'durations_ms': durations,
               'total_duration_ms': sum(durations), 'gif': settings, 'validation': game['validation'],
               'result': game['result'], 'termination': game['termination'], 'caption': game['caption'],
               'final_frame_records_termination_without_adding_a_move': True,
               'files': {paths[suffix].name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                         for suffix, data in blobs.items()}}
    blobs['json'] = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in blobs.items():
        with paths[suffix].open('xb') as stream:
            stream.write(data)
    print(json.dumps({'status': 'completed', 'accepted_moves': len(game['moves']),
                      'frames': len(original_frames), 'gif_bytes': len(gif_bytes), 'out': str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output prefix, without an extension')
    parser.add_argument('--caption', help='Plain-text provenance caption; otherwise use the trace caption')
    args = parser.parse_args()
    render(args.game, args.out, args.caption)


if __name__ == '__main__':
    main()
