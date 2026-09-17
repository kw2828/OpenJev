"""Capacity protocol, tiny CPU integration and adversarial evidence checks.

All fitting uses tiny synthetic teacher labels; no real benchmark is scored.
The native data generator and frozen secondary-grading validator remain active.
Arena games are covered by their own tests and are stubbed here.
"""

import copy
import importlib.util
import json
import math
from pathlib import Path

import chess
import chess.engine
import pytest
import torch

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_spatial_data import generate_data, validate_data

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load_script('capacity_study_test', 'chess_capacity_study.py')
grader = load_script('capacity_grader_test', 'chess_compute_study.py')
FULL_PROTOCOL = copy.deepcopy(study.PROTOCOL)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in values))


def test_full_schedules_match_across_widths_and_visit_every_example_once_per_epoch():
    starts = []
    rng_before = torch.random.get_rng_state().clone()
    for seed in (53, 67, 83):
        first = study.schedule(seed)
        assert len(first) == 6144
        assert sum(len(step['indices']) for step in first) == 98304*8 == 786432
        assert len(first)*study.PROTOCOL['depth'] == 24576
        assert [step['step'] for step in first] == list(range(1, 6145))
        for epoch in range(1, 9):
            subset = [step for step in first if step['epoch'] == epoch]
            assert len(subset) == 768
            assert all(len(step['indices']) == 128 for step in subset)
            assert sorted(i for step in subset for i in step['indices']) == list(range(98304))
        starts.append(first[0]['indices'])
        # Constructing either architecture must not alter minibatch random order.
        for width in (32, 128):
            AnchorChess('residual', seed, width, 4)
            assert study.schedule(seed) == first
    assert starts[0] != starts[1] != starts[2]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    configs = study.configs()
    assert len(configs) == 6 and configs == study.configs()
    assert {(c['width'], c['seed']) for c in configs} == {(w, s) for w in (32, 128) for s in (53, 67, 83)}
    assert all(c['name'] == f"width{c['width']}-{c['seed']}" for c in configs)


def test_architectural_capacity_counts_and_unused_head_are_explicit():
    for width, stored, active in ((32, 43726, 33185), (128, 591790, 439073)):
        model = AnchorChess('residual', 53, width, 4)
        assert model.parameter_count() == stored
        assert stored-sum(p.numel() for p in model.aux_head.parameters()) == active
    assert 'NOT matched' in study.PROTOCOL['matching']


@pytest.fixture
def signature_root(tmp_path, monkeypatch):
    root = tmp_path/'source'
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'SOURCES', ['sentinel.py'])
    for name in ('sentinel.py', study.ENGINE, 'uv.lock', 'pyproject.toml', 'old-input.json'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic frozen input\n')
    monkeypatch.setattr(study.data, 'collect_exclusions', lambda _: {
        'states': [], 'files': {'old-input.json': study.sha(root/'old-input.json')}, 'counts': {},
    })

    def forbidden(*_):
        raise AssertionError('Prepare must never generate data or train')

    monkeypatch.setattr(study.data, 'generate', forbidden)
    monkeypatch.setattr(study, 'fit', forbidden)
    return root


def test_prepare_binds_counts_schedules_openings_and_refuses_overwrite(signature_root, tmp_path):
    plan_path = study.prepare(tmp_path/'plan')
    plan = study.verify_plan(plan_path)
    assert plan['parameters'] == {'32': {'stored': 43726, 'active': 33185},
                                  '128': {'stored': 591790, 'active': 439073}}
    assert len(plan['arena_schedule']) == 96 and len(plan['openings']) == 16
    assert study.read(plan_path.parent/'prepared.json')['training_or_scoring_started'] is False
    with pytest.raises(FileExistsError):
        study.prepare(plan_path.parent)


@pytest.mark.parametrize('change', ['source', 'engine', 'exclusion', 'lock', 'plan'])
def test_frozen_input_or_budget_changes_rejected(signature_root, tmp_path, change):
    plan_path = study.prepare(tmp_path/'plan')
    if change == 'plan':
        plan = study.read(plan_path)
        plan['protocol']['epochs'] += 1
        write(plan_path, plan)
    else:
        name = {'source': 'sentinel.py', 'engine': study.ENGINE,
                'exclusion': 'old-input.json', 'lock': 'uv.lock'}[change]
        with (signature_root/name).open('a') as stream:
            stream.write('changed\n')
    with pytest.raises(ValueError, match='Frozen'):
        study.verify_plan(plan_path)


@pytest.fixture
def gate_inputs(monkeypatch):
    monkeypatch.setitem(study.PROTOCOL, 'bootstrap_replicates', 20)
    metrics, predictions, regret = {}, {}, {}
    eval_rows = {split: [{'id': f'{split}-{i}', 'game_id': f'game-{i//10}'} for i in range(100)]
                 for split in ('dev', 'shift')}
    for config in study.configs():
        name = config['name']
        large = config['width'] == 128
        correct = 40 if large else 30
        metrics[name] = {split: {'agreement': correct/100, 'target_nll': 2., 'value_mae': .4,
                                'mean_confidence': .3} for split in eval_rows}
        predictions[name] = {split: [{'correct': i < correct} for i in range(100)] for split in eval_rows}
        regret[name] = {split: .3 if large else .5 for split in eval_rows}
    return metrics, predictions, regret, eval_rows, {'gate': {'passed': True}, 'gate_passed': True}


def test_gate_requires_both_fresh_panels_and_completed_arena_lower_bound(gate_inputs):
    result = study.comparisons(*gate_inputs)
    assert result['continuation_passed'] is True
    assert len(result['checks']) == 3 and len(result['comparisons']) == 2
    for row in result['comparisons']:
        assert row['relative_reduction'] == pytest.approx(.4)
        assert row['bounded_regret_change'] == pytest.approx(-.2)
        assert row['agreement_interval']['mean'] == pytest.approx(.1)
        assert row['agreement_interval']['games'] == 10
    gate_inputs[-1]['gate']['passed'] = False
    gate_inputs[-1]['gate_passed'] = False
    assert study.comparisons(*gate_inputs)['continuation_passed'] is False


@pytest.mark.parametrize(('small', 'large', 'expected', 'passed'), [
    (.5, .4, .2, True), (.5, .45, .1, False), (.5, -.1, 1.2, True),
    (.5, .40000005, .1999999, False),
    (0., -.1, None, False), (-.1, -.2, None, False),
])
def test_relative_gate_handles_signed_scores_and_nonpositive_reference(gate_inputs, small, large, expected, passed):
    for seed in study.PROTOCOL['seeds']:
        for split in ('dev', 'shift'):
            gate_inputs[2][f'width32-{seed}'][split] = small
            gate_inputs[2][f'width128-{seed}'][split] = large
    result = study.comparisons(*gate_inputs)
    assert result['continuation_passed'] is passed
    for row in result['comparisons']:
        assert row['small_bounded_regret'] == pytest.approx(small)
        assert row['large_bounded_regret'] == pytest.approx(large)
        assert row['relative_reduction'] == (pytest.approx(expected) if expected is not None else None)


def test_one_regret_panel_failure_blocks_overall_gate(gate_inputs):
    for seed in study.PROTOCOL['seeds']:
        gate_inputs[2][f'width128-{seed}']['shift'] = .49
    result = study.comparisons(*gate_inputs)
    assert result['checks'][0]['passed'] and not result['checks'][1]['passed']
    assert not result['continuation_passed']


@pytest.mark.parametrize('change', ['fit', 'split', 'prediction', 'grade', 'extra', 'short_rows'])
def test_gate_rejects_missing_or_extra_membership(gate_inputs, change):
    metrics, predictions, regret, _, _ = gate_inputs
    if change == 'fit':
        del metrics['width128-83']
    elif change == 'split':
        del metrics['width128-83']['shift']
    elif change == 'prediction':
        del predictions['width128-83']
    elif change == 'grade':
        del regret['width128-83']
    elif change == 'extra':
        metrics['unplanned'] = copy.deepcopy(metrics['width128-83'])
    else:
        predictions['width128-83']['dev'].pop()
    with pytest.raises(ValueError):
        study.comparisons(*gate_inputs)


@pytest.fixture
def tiny_execution(tmp_path, monkeypatch):
    root = tmp_path/'source'
    protocol = copy.deepcopy(FULL_PROTOCOL)
    protocol.update(seeds=[53], train_examples=6, epochs=1, batch_size=2,
                    updates_per_fit=3, core_iterations_per_fit=12, training_device='cpu',
                    regret_positions_per_split=2, latency_positions_per_split=1, bootstrap_replicates=20)
    monkeypatch.setattr(study, 'PROTOCOL', protocol)
    monkeypatch.setattr(study, 'ROOT', root)
    engine_path = root/study.ENGINE
    engine_path.parent.mkdir(parents=True)
    engine_path.write_text('Synthetic engine marker, never executed\n')
    monkeypatch.setattr(study.data, 'collect_exclusions', lambda _: {'states': [], 'files': {}, 'counts': {}})
    fresh = copy.deepcopy(study.data.CONFIG)
    for split in fresh['splits']:
        split.update(examples=6 if split['name'] == 'train' else 4,
                     game_cap=6, max_plies=8, random_move_probability=1.0)

    def teacher(board):
        return {'target_uci': min(m.uci() for m in board.legal_moves), 'score_cp': 100, 'mate': None,
                'target_value': math.tanh(100/600), 'requested_nodes': 2000,
                'reported_nodes': 2001, 'wall_seconds': .001}

    monkeypatch.setattr(study.data, 'generate', lambda out, _engine, states: generate_data(out, teacher, fresh, states))
    monkeypatch.setattr(study.data, 'validate', lambda out, states: validate_data(out, fresh, states))
    # Source-prefixed joining is tested in test_chess_capacity_data. This fixture
    # fits six generated synthetic labels rather than loading any real old data.
    monkeypatch.setattr(study.data, 'training_rows', lambda _root, out: study.rows(out/'train.jsonl'))
    monkeypatch.setattr(study, 'signature', lambda: {
        'protocol': protocol, 'configurations': study.configs(), 'engine_sha256': study.sha(engine_path),
    })
    monkeypatch.setattr(study, 'grading_module', lambda: grader)
    events = []
    original_fit, original_evaluate = study.fit, study.evaluate

    def tracked_fit(config, *args):
        result = original_fit(config, *args)
        events.append(('fit', config['name']))
        return result

    def tracked_evaluate(*args):
        assert sum(event[0] == 'fit' for event in events) == 2
        events.append(('evaluate', None))
        return original_evaluate(*args)

    monkeypatch.setattr(study, 'fit', tracked_fit)
    monkeypatch.setattr(study, 'evaluate', tracked_evaluate)

    class Engine:
        def __init__(self):
            self.id = {'name': 'Stockfish 19 synthetic test'}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def configure(self, options):
            assert options == {'Threads': 1, 'Hash': 16} or options == {'Clear Hash': None}

        def analyse(self, board, limit, info, root_moves=None):
            assert not board.move_stack and limit.nodes == 20000 and info
            move = root_moves[0] if root_moves else min(board.legal_moves, key=lambda m: m.uci())
            assert move in board.legal_moves
            return {'pv': [move], 'score': chess.engine.PovScore(chess.engine.Cp(80 if root_moves else 100), board.turn),
                    'nodes': 20001}

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _: Engine())

    def arena_run(out, models, plan_hash):
        assert set(models) == {(32, 53), (128, 53)}
        assert all(model.recurrence == 'residual' and model.depth == 4 for model in models.values())
        assert sum(event[0] == 'evaluate' for event in events) == 4
        out.mkdir()
        write(out/'started.json', {'models': {f'width{width}-{seed}': {'state_sha256': study.digest_state(model)}
                                            for (width, seed), model in models.items()}})
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash, 'synthetic': True})
        events.append(('arena', None))

    def arena_report(out, plan_hash):
        assert study.read(out/'completed.json')['plan_sha256'] == plan_hash
        return {'gate': {'passed': False}, 'gate_passed': False, 'scope': 'Synthetic arena stub, no games played'}

    monkeypatch.setattr(study.arena, 'run', arena_run)
    monkeypatch.setattr(study.arena, 'report', arena_report)
    plan = study.prepare(tmp_path/'plan')
    execution = tmp_path/'execution'
    study.run(plan, execution)
    return plan, execution, events


def test_tiny_real_cpu_fits_preserve_unused_head_and_complete_evidence(tiny_execution, tmp_path):
    plan, execution, events = tiny_execution
    assert [kind for kind, _ in events] == ['fit', 'fit', 'evaluate', 'evaluate', 'evaluate', 'evaluate', 'arena']
    summary = study.report(plan, execution, tmp_path/'report')
    assert summary['status'] == 'completed' and len(summary['metrics']) == 2
    assert summary['novelty_established'] is False and summary['elo_estimate'] is None
    assert not summary['continuation_passed']
    assert sum(row['updates'] for row in summary['costs'].values()) == 6
    assert sum(row['core_iterations'] for row in summary['costs'].values()) == 24
    for config in study.configs():
        trained = AnchorChess.load(execution/config['name']/'weights.pt', expected_plan_sha256=study.sha(plan),
                                   expected_recurrence='residual', expected_seed=53)
        initial = AnchorChess('residual', 53, config['width'], 4)
        assert any(not torch.equal(value, initial.state_dict()[key]) for key, value in trained.state_dict().items()
                   if key.startswith('encoder.'))
        assert all(torch.equal(value, initial.state_dict()[key]) for key, value in trained.state_dict().items()
                   if key.startswith('aux_head.'))
        costs = summary['costs'][config['name']]
        assert costs['updates'] == 3 and costs['examples_seen'] == 6 and costs['core_iterations'] == 12
    with pytest.raises(FileExistsError):
        study.run(plan, execution)


def rebind(execution, name=None):
    if name:
        directory = execution/name
        trained = study.read(directory/'training.json')
        trained['learning_sha256'] = study.sha(directory/'learning.jsonl')
        write(directory/'training.json', trained)
        receipt = study.read(directory/'completed.json')
        receipt['files'] = {file: study.sha(directory/file) for file in receipt['files']}
        write(directory/'completed.json', receipt)
    complete = study.read(execution/'completed.json')
    complete['files'] = study.tree_files(execution)
    del complete['files']['completed.json']
    write(execution/'completed.json', complete)


@pytest.mark.parametrize('change', ['epoch', 'depth', 'loss', 'prediction_count', 'prediction_identity',
                                  'missing_fit_file', 'aggregate', 'latency_device', 'latency_duration',
                                  'latency_index', 'warmup_count', 'warmup_illegal'])
def test_report_rejects_semantic_corruption_despite_rebound_hashes(tiny_execution, tmp_path, change):
    plan, execution, _ = tiny_execution
    name = study.configs()[0]['name']
    directory = execution/name
    if change in ('epoch', 'depth', 'loss'):
        values = study.rows(directory/'learning.jsonl')
        values[0][change] += 1
        write_rows(directory/'learning.jsonl', values)
    elif change.startswith('prediction_'):
        values = study.rows(directory/'dev.jsonl')
        if change == 'prediction_count':
            values.pop()
        else:
            values[0]['id'] = 'different-position'
        write_rows(directory/'dev.jsonl', values)
    elif change in ('missing_fit_file', 'aggregate'):
        receipt = study.read(directory/'completed.json')
        if change == 'missing_fit_file':
            del receipt['files']['dev.jsonl']
            (directory/'dev.jsonl').unlink()
        else:
            receipt['evaluation']['dev']['metrics']['agreement'] += .1
        write(directory/'completed.json', receipt)
    else:
        latency = study.read(directory/'latency.json')
        if change == 'latency_device':
            latency['device'] = 'mps'
        elif change == 'latency_duration':
            latency['records'][0]['wall_ms'] = -1.
            latency['total_wall_ms'] = sum(row['wall_ms'] for row in latency['records'])
        elif change == 'latency_index':
            latency['records'][0]['index'] += 1
        elif change == 'warmup_count':
            latency['warmup_records'].pop()
        else:
            latency['warmup_records'][0]['choice'] = 'a1a8'
        write(directory/'latency.json', latency)
    rebind(execution, name)
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path/'corrupt-report')


@pytest.mark.parametrize('seconds', [-1., math.nan, math.inf, -math.inf])
def test_nonfinite_or_negative_evaluation_duration_rejected(tiny_execution, tmp_path, seconds):
    plan, execution, _ = tiny_execution
    name = study.configs()[0]['name']
    path = execution/name/'completed.json'
    receipt = study.read(path)
    receipt['evaluation']['dev']['evaluation_wall_seconds'] = seconds
    path.write_text(json.dumps(receipt, allow_nan=True)+'\n')
    rebind(execution)
    with pytest.raises(ValueError, match='Evaluation aggregate or time'):
        study.report(plan, execution, tmp_path/'corrupt-time-report')


def test_failed_receipt_is_terminal_and_never_silently_restarted(tiny_execution, tmp_path, monkeypatch):
    plan, _, _ = tiny_execution

    def fail(*_):
        raise RuntimeError('Synthetic generator failure')

    monkeypatch.setattr(study.data, 'generate', fail)
    out = tmp_path/'failed-execution'
    with pytest.raises(RuntimeError, match='generator failure'):
        study.run(plan, out)
    assert study.read(out/'failed.json')['status'] == 'failed'
    assert study.read(out/'failed.json')['plan_sha256'] == study.sha(plan)
    assert not (out/'completed.json').exists()
    with pytest.raises(FileExistsError):
        study.run(plan, out)


def test_arena_must_bind_the_exact_trained_weights_even_with_rehashed_outer_receipt(tiny_execution, tmp_path):
    plan, execution, _ = tiny_execution
    path = execution/'arena/started.json'
    receipt = study.read(path)
    receipt['models']['width128-53']['state_sha256'] = '0'*64
    write(path, receipt)
    rebind(execution)
    with pytest.raises(ValueError, match='(?i)arena.*(model|checkpoint|weight|binding)'):
        study.report(plan, execution, tmp_path/'wrong-arena-model')
