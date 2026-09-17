"""Tiny fake-teacher fixtures only; no engine binary or actual training data."""

import copy
import hashlib
import json
import math

import chess
import chess.engine
import pytest

from openjev.research import chess_capacity_data as data
from openjev.research.chess_spatial_data import generate_data, state_key, symmetry_key


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def small_config(source, examples=2, seed_base=1000):
    config = copy.deepcopy(source)
    for index, split in enumerate(config['splits']):
        split.update(examples=examples, game_cap=8, max_plies=12,
                     random_move_probability=1., seed_base=seed_base+100*index)
    return config


def fake_label(board):
    return {'target_uci': min(move.uci() for move in board.legal_moves), 'score_cp': 100, 'mate': None,
            'target_value': math.tanh(100/600), 'requested_nodes': 2000,
            'reported_nodes': 2001, 'wall_seconds': .001}


class FakeEngine:
    def __init__(self, name='Stockfish 19 synthetic test'):
        self.id = {'name': name}
        self.calls = 0
        self.configured = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def configure(self, values):
        self.configured.append(values)

    def analyse(self, board, limit, info):
        assert not board.move_stack and limit.nodes == 2000
        self.calls += 1
        return {'pv': [min(board.legal_moves, key=lambda move: move.uci())],
                'score': chess.engine.PovScore(chess.engine.Cp(100), board.turn), 'nodes': 2001}


def fixture_sources(tmp_path, monkeypatch):
    old_config = small_config(data.CONFIG, seed_base=1000)
    monkeypatch.setattr(data, 'ORIGINAL_COUNT', 2)
    original = (tmp_path/data.TRAIN).parent
    generate_data(original, fake_label, old_config, set())
    states, files, by_file = set(), {}, {}
    for split in ('train', 'dev', 'shift'):
        relative = str((original/f'{split}.jsonl').relative_to(tmp_path))
        files[relative] = sha(tmp_path/relative)
        rows = read_rows(tmp_path/relative)
        for row in rows:
            states.add(state_key(chess.Board(row['fen'])))
            states.add(state_key(chess.Board(row['next_fen'])))
            states.update(state_key(chess.Board(t['next_fen'])) for t in row['aux_transitions'])
        by_file[relative] = {'kind': 'dataset', 'rows': 2, 'input_positions': 2,
                             'behavior_successor_positions': 2, 'auxiliary_successor_positions': 8,
                             'positions': 12}
    inherited = {'states': sorted(states), 'files': files,
                 'counts': {'source_files': 3, 'observed_positions': 36, 'unique_state_keys': len(states),
                            'unique_mirror_classes': len({symmetry_key(chess.Board(key+' 0 1')) for key in states}),
                            'dataset_rows': 6, 'dataset_input_positions': 6,
                            'dataset_behavior_successor_positions': 6,
                            'dataset_auxiliary_successor_positions': 24, 'by_file': by_file}}
    monkeypatch.setattr(data.anchor_source, 'collect_exclusions', lambda root: copy.deepcopy(inherited))
    anchor_config = small_config(data.ANCHOR_CONFIG, seed_base=3000)
    monkeypatch.setattr(data, 'ANCHOR_CONFIG', anchor_config)
    anchor = tmp_path/data.ANCHOR_EXECUTION
    generate_data(anchor/'data', fake_label, anchor_config, states)
    write_json(anchor/'completed.json', {'status': 'completed',
                                         'data_receipt_sha256': sha(anchor/'data/completed.json')})
    config = small_config(data.CONFIG, seed_base=6000)
    config['splits'][0]['examples'] = 4
    monkeypatch.setattr(data, 'CONFIG', config)
    return tmp_path, inherited


def rebind_anchor(root, member):
    directory = root/data.ANCHOR_EXECUTION
    receipt = json.loads((directory/'data/completed.json').read_text())
    receipt['files'][member] = sha(directory/'data'/member)
    write_json(directory/'data/completed.json', receipt)
    complete = json.loads((directory/'completed.json').read_text())
    complete['data_receipt_sha256'] = sha(directory/'data/completed.json')
    write_json(directory/'completed.json', complete)


def test_config_exact_fixed_quotas_and_teacher_budget():
    assert data.ORIGINAL_COUNT == 32768
    assert data.CONFIG['teacher_nodes'] == 2000
    assert data.CONFIG['teacher_threads'] == 1 and data.CONFIG['teacher_hash_mb'] == 16
    assert data.CONFIG['value_cp_scale'] == 600 and data.CONFIG['mate_cp'] == 10000
    assert data.CONFIG['aux_actions'] == 4
    assert [(s['name'], s['examples'], s['game_cap'], s['max_plies'], s['random_move_probability'], s['seed_base'])
            for s in data.CONFIG['splits']] == [
        ('train', 65536, 2400, 64, .5, 105000000),
        ('dev', 4096, 300, 64, .5, 106000000),
        ('shift', 4096, 300, 96, .1, 107000000),
    ]


def test_exclusions_bind_every_anchor_member_and_all_successors(tmp_path, monkeypatch):
    root, inherited = fixture_sources(tmp_path, monkeypatch)
    result = data.collect_exclusions(root)
    assert set(result) == {'states', 'files', 'counts'}
    assert set(inherited['states']) <= set(result['states'])
    for split in ('dev', 'shift'):
        for row in read_rows(root/data.ANCHOR_EXECUTION/'data'/f'{split}.jsonl'):
            for fen in [row['fen'], row['next_fen']]+[item['next_fen'] for item in row['aux_transitions']]:
                assert state_key(chess.Board(fen)) in result['states']
    assert result['counts']['anchor_input_positions'] == 4
    assert result['counts']['anchor_behavior_successor_positions'] == 4
    assert result['counts']['anchor_auxiliary_successor_positions'] == 16
    assert result['counts']['observed_positions'] == 60
    assert result['counts']['dataset_rows'] == 10
    assert result['counts']['unique_state_keys'] == len(result['states'])
    assert result['counts']['unique_mirror_classes'] == len({symmetry_key(chess.Board(k+' 0 1')) for k in result['states']})
    assert result['counts']['source_files'] == len(result['files']) == len(result['counts']['by_file'])
    for path, digest in result['files'].items():
        assert digest == sha(root/path)
    assert f'{data.ANCHOR_EXECUTION}/completed.json' in result['files']
    assert f'{data.ANCHOR_EXECUTION}/data/analyses.jsonl' in result['files']
    assert 'target_uci' not in json.dumps(result) and 'target_value' not in json.dumps(result)


@pytest.mark.parametrize('corruption', ['missing_member', 'wrong_hash', 'extra_receipt_member',
                                     'missing_receipt_member', 'wrong_outer_binding', 'failed_outer',
                                     'failed_data', 'wrong_counts', 'wrong_config', 'symlink'])
def test_anchor_receipt_or_membership_corruption_rejected(tmp_path, monkeypatch, corruption):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    directory = root/data.ANCHOR_EXECUTION
    member = directory/'data/dev.jsonl'
    if corruption == 'missing_member':
        member.unlink()
    elif corruption == 'wrong_hash':
        member.write_bytes(member.read_bytes()+b'\n')
    elif corruption == 'symlink':
        replacement = root/'replacement.jsonl'
        replacement.write_bytes(member.read_bytes())
        member.unlink()
        member.symlink_to(replacement)
    elif corruption == 'wrong_config':
        record = json.loads((directory/'data/started.json').read_text())
        record['config']['teacher_nodes'] = 1
        write_json(directory/'data/started.json', record)
        rebind_anchor(root, 'started.json')
    elif corruption in ('wrong_outer_binding', 'failed_outer'):
        receipt = json.loads((directory/'completed.json').read_text())
        receipt['data_receipt_sha256' if corruption == 'wrong_outer_binding' else 'status'] = 'wrong'
        write_json(directory/'completed.json', receipt)
    else:
        receipt = json.loads((directory/'data/completed.json').read_text())
        if corruption == 'extra_receipt_member':
            receipt['files']['unexpected.json'] = '0'*64
        elif corruption == 'missing_receipt_member':
            receipt['files'].pop('analyses.jsonl')
        elif corruption == 'failed_data':
            write_json(directory/'data/failed.json', {'status': 'failed'})
        else:
            receipt['counts']['dev'] = 1
        write_json(directory/'data/completed.json', receipt)
        outer = json.loads((directory/'completed.json').read_text())
        outer['data_receipt_sha256'] = sha(directory/'data/completed.json')
        write_json(directory/'completed.json', outer)
    with pytest.raises(ValueError):
        data.collect_exclusions(root)


@pytest.mark.parametrize('corruption', ['missing_behavior', 'wrong_behavior', 'missing_auxiliary',
                                     'duplicate_action', 'illegal_action', 'wrong_successor',
                                     'wrong_split', 'wrong_id', 'wrong_state', 'missing_row'])
def test_rehashed_anchor_transition_corruption_rejected(tmp_path, monkeypatch, corruption):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    path = root/data.ANCHOR_EXECUTION/'data/dev.jsonl'
    rows = read_rows(path)
    row = rows[0]
    if corruption == 'missing_behavior':
        row.pop('behavior_uci')
    elif corruption == 'wrong_behavior':
        row['next_fen'] = row['fen']
    elif corruption == 'missing_auxiliary':
        row['aux_transitions'].pop()
    elif corruption == 'duplicate_action':
        row['aux_transitions'][1] = row['aux_transitions'][0].copy()
    elif corruption == 'illegal_action':
        row['aux_transitions'][0]['uci'] = '0000'
    elif corruption == 'wrong_successor':
        row['aux_transitions'][0]['next_fen'] = row['fen']
    elif corruption == 'wrong_split':
        row['split'] = 'train'
    elif corruption == 'wrong_id':
        row['id'] = 'dev-99999'
    elif corruption == 'wrong_state':
        row['state_key'] = 'wrong'
    else:
        rows.pop()
    write_rows(path, rows)
    rebind_anchor(root, 'dev.jsonl')
    with pytest.raises(ValueError):
        data.collect_exclusions(root)


def test_generation_validation_and_join_are_disjoint_and_preserve_sources(tmp_path, monkeypatch):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    exclusions = data.collect_exclusions(root)
    # Supplying only a mirror must still exclude the natural starting state.
    states = set(exclusions['states']) | {state_key(chess.Board().mirror())}
    engine = FakeEngine()
    monkeypatch.setattr(data.chess.engine.SimpleEngine, 'popen_uci', lambda _: engine)
    out = root/'fresh'
    receipt = data.generate(out, 'fake-engine-only', states)
    validated = data.validate(out, states)
    assert receipt['counts'] == {'train': 4, 'dev': 2, 'shift': 2}
    assert receipt['teacher_calls'] == engine.calls > 8
    assert receipt['requested_nodes'] == engine.calls*2000
    assert receipt['reported_nodes'] == engine.calls*2001
    assert engine.configured[0] == {'Threads': 1, 'Hash': 16}
    assert engine.configured[1:] == [{'Clear Hash': None}]*engine.calls
    excluded_classes = {symmetry_key(chess.Board(key+' 0 1')) for key in states}
    seen = set()
    games = []
    for rows in validated.values():
        games.append({row['game_id'] for row in rows})
        for row in rows:
            key = symmetry_key(chess.Board(row['fen']))
            assert key not in seen and key not in excluded_classes
            seen.add(key)
    assert all(left.isdisjoint(right) for i, left in enumerate(games) for right in games[i+1:])
    before = (root/data.TRAIN).read_bytes(), (out/'train.jsonl').read_bytes()
    joined = data.training_rows(root, out)
    assert len(joined) == 6 and len({row['id'] for row in joined}) == 6
    assert [row['id'] for row in joined[:2]] == ['old:train-00000', 'old:train-00001']
    assert joined[2]['id'] == 'fresh:train-00000'
    assert all(row['game_id'] == f'{row["training_source"]}:{row["source_game_id"]}' for row in joined)
    assert all(type(row['source_game_id']) is int for row in joined)
    assert before == ((root/data.TRAIN).read_bytes(), (out/'train.jsonl').read_bytes())
    for source, records in (('old', read_rows(root/data.TRAIN)), ('fresh', validated['train'])):
        for actual, expected in zip([r for r in joined if r['training_source'] == source], records, strict=True):
            assert actual['target_uci'] == expected['target_uci']
            assert actual['target_value'] == expected['target_value']
            assert actual['fen'] == expected['fen']
    with pytest.raises(FileExistsError):
        data.generate(out, 'fake-engine-only', states)


def test_join_rejects_rehashed_cross_source_mirror_duplicate(tmp_path, monkeypatch):
    root, _ = fixture_sources(tmp_path, monkeypatch)
    exclusions = data.collect_exclusions(root)
    out = root/'fresh'
    generate_data(out, fake_label, data.CONFIG, exclusions['states'])
    rows = read_rows(out/'train.jsonl')
    old = read_rows(root/data.TRAIN)[0]
    mirror = chess.Board(old['fen']).mirror()
    rows[0].update(fen=mirror.fen(en_passant='fen'), state_key=state_key(mirror),
                   target_uci=min(move.uci() for move in mirror.legal_moves))
    write_rows(out/'train.jsonl', rows)
    receipt = json.loads((out/'completed.json').read_text())
    receipt['files']['train.jsonl'] = sha(out/'train.jsonl')
    write_json(out/'completed.json', receipt)
    with pytest.raises(ValueError, match='mirrored'):
        data.training_rows(root, out)


def test_generate_honors_mirror_only_exclusion_and_global_split_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'CONFIG', small_config(data.CONFIG, seed_base=8000))
    engine = FakeEngine()
    monkeypatch.setattr(data.chess.engine.SimpleEngine, 'popen_uci', lambda _: engine)
    excluded = {state_key(chess.Board().mirror())}
    assert state_key(chess.Board()) not in excluded
    out = tmp_path/'mirror-only'
    data.generate(out, 'fake-engine-only', excluded)
    rows = data.validate(out, excluded)
    classes = [symmetry_key(chess.Board(row['fen'])) for split in rows.values() for row in split]
    assert len(classes) == len(set(classes)) == 6
    assert symmetry_key(chess.Board()) not in classes


def test_failed_generation_is_not_retried_and_retains_failure_receipt(tmp_path, monkeypatch):
    config = small_config(data.CONFIG)
    monkeypatch.setattr(data, 'CONFIG', config)

    class FailingEngine(FakeEngine):
        def analyse(self, board, limit, info):
            self.calls += 1
            raise RuntimeError('synthetic failure')
    engine = FailingEngine()
    monkeypatch.setattr(data.chess.engine.SimpleEngine, 'popen_uci', lambda _: engine)
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='synthetic failure'):
        data.generate(out, 'fake-engine-only', set())
    assert engine.calls == 1 and (out/'failed.json').exists()
    assert not (out/'completed.json').exists()
    with pytest.raises(FileExistsError):
        data.generate(out, 'fake-engine-only', set())
    assert engine.calls == 1


def test_wrong_engine_version_fails_before_any_label(tmp_path, monkeypatch):
    engine = FakeEngine('Stockfish 18 synthetic test')
    monkeypatch.setattr(data.chess.engine.SimpleEngine, 'popen_uci', lambda _: engine)
    with pytest.raises(ValueError, match='Stockfish 19'):
        data.generate(tmp_path/'not-created', 'fake-engine-only', set())
    assert engine.calls == 0 and not (tmp_path/'not-created').exists()
