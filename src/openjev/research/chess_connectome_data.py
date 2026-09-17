"""Fresh connectome-study panels with inherited and full ChessBench exclusions.

Only board identities from earlier work enter exclusion construction. The new
panels reuse the audited candidate generator through an isolated module object,
with explicit new seeds; the original module and its frozen CONFIG are untouched.
"""

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import chess
import chess.engine

from openjev.research import chess_candidate_data as candidate
from openjev.research.chess_spatial_data import state_key
from openjev.research.chessbench_data import decode_bc_bag

TRANSFER_PLAN = 'evidence/chessbench-transfer-v1/protocol/plan.json'
TRANSFER_PLAN_SHA = '10c280ddd417965b2a24aef85d564b34e0a5b171138d416e880327a2c2304323'
TRANSFER_PUBLICATION = 'evidence/chessbench-transfer-v1/results'
TRANSFER_ARCHIVE_SHA = '2e109ce9d4e088ce060e86d67fe5b6ec9274a6f3c80acdf31973e1a5e9ac2656'
TRAIN = candidate.TRAIN
CONFIG = copy.deepcopy(candidate.CONFIG)
CONFIG['splits'][0]['seed_base'] = 115000000
CONFIG['splits'][1]['seed_base'] = 116000000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def generator(config=CONFIG):
    """An isolated instance of the unchanged, strict generator and validator."""
    path = Path(candidate.__file__)
    spec = importlib.util.spec_from_file_location('_connectome_candidate_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = copy.deepcopy(config)
    return module


def add_exposures(states, fens):
    """Extend a natural-state set by every distinct root and all legal children."""
    seen, children, roots = set(), 0, 0
    for fen in fens:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError('Invalid prior board')
        key = state_key(board)
        if key in seen:
            continue
        seen.add(key)
        states.add(key)
        roots += 1
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            states.add(state_key(child))
            children += 1
    return {'distinct_roots': roots, 'native_successor_visits': children}


def collect_exclusions(root):
    root = Path(root)
    plan_path = root/TRANSFER_PLAN
    if sha(plan_path) != TRANSFER_PLAN_SHA:
        raise ValueError('Committed transfer plan changed')
    plan = read(plan_path)
    publication = root/TRANSFER_PUBLICATION
    complete = read(publication/'completed.json')
    if (complete['status'] != 'completed' or complete['plan_sha256'] != TRANSFER_PLAN_SHA
            or complete['files']['chessbench-transfer-v1.tar.gz'] != TRANSFER_ARCHIVE_SHA):
        raise ValueError('Transfer publication is not the completed pinned package')
    for name, digest in complete['files'].items():
        if sha(publication/name) != digest:
            raise ValueError('Published transfer evidence changed')
    snapshot = plan_path.parent/plan['exclusion_file']
    if sha(snapshot) != plan['exclusion_sha256']:
        raise ValueError('Inherited exclusion snapshot changed')
    raw = gzip.decompress(snapshot.read_bytes())
    if hashlib.sha256(raw).hexdigest() != plan['exclusion_uncompressed_sha256']:
        raise ValueError('Inherited exclusion contents changed')
    inherited = json.loads(raw)
    if inherited != sorted(set(inherited)) or len(inherited) != plan['exposure']['total_natural_states']:
        raise ValueError('Inherited exclusion membership differs')
    source = root/plan['input']
    if sha(source) != plan['input_sha256']:
        raise ValueError('ChessBench source bytes changed')
    decoded = decode_bc_bag(source)
    states = set(inherited)
    exposure = add_exposures(states, (row['fen'] for row in decoded))
    files = {TRANSFER_PLAN: TRANSFER_PLAN_SHA,
             snapshot.relative_to(root).as_posix(): plan['exclusion_sha256'],
             plan['input']: plan['input_sha256'],
             f'{TRANSFER_PUBLICATION}/completed.json': sha(publication/'completed.json')}
    files.update({f'{TRANSFER_PUBLICATION}/{name}': digest for name, digest in complete['files'].items()})
    return {'states': sorted(states), 'files': dict(sorted(files.items())),
            'counts': {'inherited_natural_states': len(inherited), 'chessbench_source_rows': len(decoded),
                       **exposure, 'total_natural_states': len(states)},
            'scope': 'Inherited transfer exclusions plus ALL decoded ChessBench roots and legal successors; both mirrors excluded by admission. No historical labels enter new inputs.'}


def generate(out, engine, states):
    if Path(out).exists():
        raise FileExistsError(out)
    module = generator()
    with chess.engine.SimpleEngine.popen_uci(str(engine)) as teacher:
        if not teacher.id.get('name', '').startswith('Stockfish 19'):
            raise ValueError('Expected Stockfish 19')
        teacher.configure({'Threads': CONFIG['teacher_threads'], 'Hash': CONFIG['teacher_hash_mb']})
        return module._generate(out, lambda board: module.native.stockfish_label(teacher, board, CONFIG), CONFIG, states)


def validate(out, states):
    return generator().validate(out, states)


def training_rows(root):
    return candidate.training_rows(root)
