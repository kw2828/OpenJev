"""Synthetic-only study mechanics; no released checkpoint or scored panel is used."""

import copy
import importlib.util
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import pytest
import torch

from openjev.research.chess_spatial import MODES, SpatialChess

ROOT = Path(__file__).resolve().parents[1]


def load_study():
    spec = importlib.util.spec_from_file_location('compute_study_test', ROOT/'scripts/chess_compute_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


class SyntheticEngine:
    def __init__(self):
        self.id = {'name': 'Stockfish 19 synthetic unit-test engine'}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def configure(self, _options):
        pass

    def analyse(self, board, limit, *, info=None, root_moves=None):
        legal = list(board.legal_moves) if root_moves is None else root_moves
        move = min(legal, key=lambda m: m.uci())
        # Deliberately creates negative finite-search differences to test retention.
        cp = 0 if root_moves is None else 30
        return {'pv': [move], 'score': chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
                'nodes': limit.nodes+1}


def original_fixture(study, directory):
    engine = directory/'fake-engine'
    engine.write_text('Unit-test stub, not an executable engine.\n')
    original_plan = directory/'original-plan.json'
    write(original_plan, {'sources': {'src/openjev/research/chess_spatial.py':
                                      study.sha(ROOT/'src/openjev/research/chess_spatial.py')},
                          'engine': {'path': str(engine)}})
    execution = directory/'original-execution'
    files = {}
    for split in ('dev', 'shift'):
        rows = []
        board = chess.Board()
        sequence = ['e2e4', 'e7e5', 'g1f3', 'b8c6'] if split == 'dev' else ['d2d4', 'd7d5', 'c2c4', 'e7e6']
        for index, action in enumerate(sequence):
            board.push_uci(action)
            rows.append({'id': f'{split}-{index}', 'game_id': index,
                         'fen': board.fen(en_passant='fen'),
                         'target_uci': min(move.uci() for move in board.legal_moves)})
        file = execution/'data'/f'{split}.jsonl'
        write_rows(file, rows)
        files[file.name] = study.sha(file)
    write(execution/'data/completed.json', {'status': 'completed', 'files': files})
    receipts = {}
    for mode in MODES:
        name = f'{mode}-17'
        fit = execution/name
        fit.mkdir()
        SpatialChess(mode, 17, width=4).save(fit/'weights.pt', plan_sha256=study.sha(original_plan))
        write(fit/'completed.json', {
            'status': 'completed', 'mode': mode, 'seed': 17,
            'plan_sha256': study.sha(original_plan),
            'data_receipt_sha256': study.sha(execution/'data/completed.json'),
            'files': {'weights.pt': study.sha(fit/'weights.pt')},
        })
        receipts[name] = study.sha(fit/'completed.json')
    write(execution/'completed.json', {'status': 'completed', 'plan_sha256': study.sha(original_plan),
                                      'fits': 4, 'fit_receipts': receipts})
    return original_plan, execution, engine


@pytest.fixture(scope='module')
def synthetic_run(tmp_path_factory):
    study = load_study()
    study.PROTOCOL = copy.deepcopy(study.PROTOCOL)
    study.PROTOCOL.update({'seeds': [17], 'positions_per_split': 2,
                           'regret_positions_per_split': 1, 'warmups_per_configuration': 1})
    directory = tmp_path_factory.mktemp('compute-synthetic')
    original, source, engine = original_fixture(study, directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _path: SyntheticEngine())
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        plan = study.prepare(directory/'plan', original, source, engine)
        study.run(plan, directory/'execution')
        summary = study.report(plan, directory/'execution', directory/'report')
        yield study, plan, directory, summary
    finally:
        patch.undo()
        torch.set_num_threads(previous)


def test_exact_90_configuration_panel_with_no_seed_selection():
    study = load_study()
    configs = study.configurations()
    assert len(configs) == len({c['id'] for c in configs}) == 90
    counts = Counter(c['mode'] for c in configs)
    assert counts == {'cnn': 18, 'recurrent': 24, 'reconstruct': 24, 'predict': 24}
    for mode in MODES:
        for seed in (17, 29, 43):
            cells = [c for c in configs if c['mode'] == mode and c['seed'] == seed]
            assert len(cells) == (6 if mode == 'cnn' else 8)
            assert {c['top_k'] for c in cells if c['method'] == 'successor'} == {None, 2, 4}
    assert configs == study.configurations()


def test_synthetic_end_to_end_has_full_shared_panels_warmups_and_negative_secondary(synthetic_run):
    study, plan_path, directory, summary = synthetic_run
    plan = study.read(plan_path)
    assert summary['status'] == 'completed' and len(summary['configurations']) == 60
    assert summary['warmups']['calls'] == 30
    assert all(r['examples'] == 2 and r['secondary_positions'] == 1 for r in summary['configurations'])
    assert all(r['mean_bounded_regret'] < 0 and r['mean_cp_loss'] == -30 for r in summary['configurations'])
    assert summary['novelty_established'] is False and summary['elo_estimate'] is None
    decisions = study.jsonl(directory/'execution/decisions.jsonl')
    assert len(decisions) == 120
    for split in ('dev', 'shift'):
        expected = [row['id'] for row in plan['panels'][split]]
        for config in plan['configurations']:
            assert [r['id'] for r in decisions if r['configuration'] == config['id'] and r['split'] == split] == expected
    for row in decisions:
        if '-successor_2_' in row['configuration']:
            assert row['policy_entropy'] is not None and row['target_probability'] is not None
            assert row['compute']['policy_forward_calls'] == 1
        if '-successor_all_' in row['configuration']:
            assert row['policy_entropy'] is None and row['value'] is None
        assert 'scores' not in row and 'probabilities' not in row
    with pytest.raises(FileExistsError):
        study.run(plan_path, directory/'execution')


def test_gate_requires_both_panels_and_every_configuration():
    study = load_study()
    rows = [{'configuration': c['id'], 'split': split,
             'agreement': .4 if c['mode'] == 'predict' and c['variant'] == 'policy_d8' else .3}
            for c in study.configurations() for split in ('dev', 'shift')]
    assert study.gate(rows)['passed']
    with pytest.raises(ValueError, match='complete fixed'):
        study.gate(rows[:-1])
    with pytest.raises(ValueError, match='complete fixed'):
        study.gate(rows+[rows[0]])
    next(r for r in rows if r['configuration'] == 'predict-17-policy_d8' and r['split'] == 'shift')['agreement'] = .28
    assert not study.gate(rows)['passed']


def test_paired_corrected_and_broken_preserve_net_change(synthetic_run):
    _, _, _, summary = synthetic_run
    lookup = {(r['configuration'], r['split']): r for r in summary['configurations']}
    for row in summary['configurations']:
        base = lookup[f"{row['mode']}-{row['seed']}-policy_d4", row['split']]
        paired = row['paired_vs_depth4']
        assert paired['net_agreement_gain'] == pytest.approx(row['agreement']-base['agreement'])
        assert paired['net_agreement_gain'] == (paired['corrected']-paired['broken'])/row['examples']


def copy_execution(synthetic_run, tmp_path):
    study, plan, directory, _ = synthetic_run
    execution = tmp_path/'execution'
    shutil.copytree(directory/'execution', execution)
    return study, plan, execution


def rehash(study, execution, name):
    complete = study.read(execution/'completed.json')
    complete['files'][name] = study.sha(execution/name)
    write(execution/'completed.json', complete)


def test_hash_change_is_rejected(synthetic_run, tmp_path):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    with (execution/'decisions.jsonl').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash'):
        study.report(plan, execution, tmp_path/'report')


@pytest.mark.parametrize('kind', ['illegal', 'correctness', 'cost', 'confidence', 'duplicate', 'shortlist'])
def test_rehashed_invalid_decision_is_rejected(synthetic_run, tmp_path, kind):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    rows = study.jsonl(execution/'decisions.jsonl')
    row = next(r for r in rows if '-successor_2_' in r['configuration'])
    if kind == 'illegal':
        row['choice'] = 'e2e5'
    elif kind == 'correctness':
        row['correct'] = not row['correct']
    elif kind == 'cost':
        row['compute']['boardvalue_evaluations'] += 1
    elif kind == 'confidence':
        row['policy_max_probability'] = 1.5
    elif kind == 'duplicate':
        rows[1] = rows[0]
    else:
        row['evaluated_candidates'] = row['evaluated_candidates'][:1]
    write_rows(execution/'decisions.jsonl', rows)
    rehash(study, execution, 'decisions.jsonl')
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path/'report')


@pytest.mark.parametrize('kind', ['analysis_score', 'regret', 'missing_call', 'wrong_choice', 'cost'])
def test_rehashed_secondary_corruption_is_rejected(synthetic_run, tmp_path, kind):
    study, plan, execution = copy_execution(synthetic_run, tmp_path)
    if kind == 'cost':
        complete = study.read(execution/'completed.json')
        complete['secondary_cost']['reported_nodes'] += 1
        write(execution/'completed.json', complete)
    elif kind in ('analysis_score', 'missing_call'):
        rows = study.jsonl(execution/'analyses.jsonl')
        if kind == 'analysis_score':
            rows[0]['bounded_score'] += .5
        else:
            rows.pop()
        write_rows(execution/'analyses.jsonl', rows)
        rehash(study, execution, 'analyses.jsonl')
    else:
        rows = study.jsonl(execution/'regret.jsonl')
        if kind == 'regret':
            rows[0]['bounded_regret'] *= -1
        else:
            rows[0]['choice'] = 'e2e5'
        write_rows(execution/'regret.jsonl', rows)
        rehash(study, execution, 'regret.jsonl')
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path/'report')


def test_failure_keeps_partial_journals_and_cannot_retry(synthetic_run, tmp_path, monkeypatch):
    study, plan, _, _ = synthetic_run
    original = study.decide
    calls = 0

    def fail_once(model, board, config):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('synthetic failure, no replacement')
        return original(model, board, config)

    monkeypatch.setattr(study, 'decide', fail_once)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        study.run(plan, tmp_path/'failed-execution')
    failed = tmp_path/'failed-execution'
    assert study.read(failed/'failed.json')['status'] == 'failed'
    assert len(study.jsonl(failed/'warmups.jsonl')) == 1
    assert len(study.jsonl(failed/'decisions.jsonl')) == 1
    assert not (failed/'completed.json').exists()
    with pytest.raises(FileExistsError):
        study.run(plan, failed)


def test_frozen_selection_and_configuration_changes_are_rejected(synthetic_run, tmp_path):
    study, plan, _, _ = synthetic_run
    edited = study.read(plan)
    edited['secondary_indices']['dev'] = [1-edited['secondary_indices']['dev'][0]]
    changed = tmp_path/'changed-plan.json'
    write(changed, edited)
    with pytest.raises(ValueError, match='Frozen'):
        study.verify_plan(changed)


@pytest.mark.parametrize('mate', [2, -3, 0])
def test_secondary_mate_scores_preserve_signed_distance_including_zero(synthetic_run, mate):
    study, plan_path, directory, _ = synthetic_run
    plan = study.read(plan_path)
    execution = directory/'execution'
    decisions = study.jsonl(execution/'decisions.jsonl')
    analyses = study.jsonl(execution/'analyses.jsonl')
    records = study.jsonl(execution/'regret.jsonl')
    cost = study.read(execution/'completed.json')['secondary_cost']
    changed = analyses[0]
    changed['mate'] = mate
    changed['score_cp'] = chess.engine.Mate(mate).score(mate_score=study.PROTOCOL['mate_cp'])
    changed['bounded_score'] = math.tanh(changed['score_cp']/study.PROTOCOL['value_cp_scale'])
    lookup = {(r['split'], r['panel_index'], r['root_move']): r for r in analyses}
    for record in records:
        best = lookup[record['split'], record['panel_index'], None]
        chosen = lookup[record['split'], record['panel_index'], record['choice']]
        record['bounded_regret'] = best['bounded_score']-chosen['bounded_score']
        record['cp_loss'] = best['score_cp']-chosen['score_cp']
    study.validate_secondary(plan, decisions, analyses, records, cost)
    changed['score_cp'] += 1
    changed['bounded_score'] = math.tanh(changed['score_cp']/study.PROTOCOL['value_cp_scale'])
    with pytest.raises(ValueError, match='secondary engine analysis'):
        study.validate_secondary(plan, decisions, analyses, records, cost)


def test_secondary_timing_includes_clear_hash_overhead(synthetic_run, tmp_path, monkeypatch):
    study, plan_path, directory, _ = synthetic_run
    events = []
    timer_count = 0

    class InstrumentedEngine(SyntheticEngine):
        def configure(self, options):
            if 'Clear Hash' in options:
                events.append('clear_hash')

        def analyse(self, *args, **kwargs):
            events.append('analyse')
            return super().analyse(*args, **kwargs)

    def timer():
        nonlocal timer_count
        events.append('timer')
        timer_count += 1
        return timer_count*.001

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _path: InstrumentedEngine())
    monkeypatch.setattr(study.time, 'perf_counter', timer)
    costs = study.score_secondary(study.read(plan_path), tmp_path,
                                  study.jsonl(directory/'execution/decisions.jsonl'))
    assert events == ['timer', 'clear_hash', 'analyse', 'timer']*costs['calls']
    assert costs['wall_seconds'] == pytest.approx(.001*costs['calls'])
