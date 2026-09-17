"""Synthetic mechanics and receipt checks; no real models or benchmark scoring."""

import copy
import importlib.util
import json
import math
import random
from pathlib import Path

import chess
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_study():
    spec = importlib.util.spec_from_file_location('_mate_study_test', ROOT/'scripts/chess_mate_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study():
    return load_study()


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def row(name='ordinary', board=None):
    board = chess.Board() if board is None else board
    return {'id': name, 'fen': board.fen(en_passant='fen'),
            'target_uci': min(move.uci() for move in board.legal_moves), 'target_value': .25}


def mate_row(name='mate'):
    # Artificial three-piece fixture, never a benchmark position.
    board = chess.Board('7k/5K2/6Q1/8/8/8/8/8 w - - 0 1')
    moves = []
    for move in board.legal_moves:
        successor = board.copy()
        successor.push(move)
        if successor.is_checkmate():
            moves.append(move.uci())
    assert len(moves) > 1
    moves.sort()
    return {'id': name, 'fen': board.fen(), 'target_uci': moves[0], 'mating_uci': moves,
            'source_game': 'https://lichess.org/Fixture1#2'}


def test_set_loss_is_negative_log_total_winning_probability_and_replay_stays_ce(study):
    logits = torch.tensor([[.2, .3, .5, 0.], [.1, .6, .3, 0.]], dtype=torch.float64).log()
    targets = torch.tensor([0, 1])
    mating = torch.tensor([[True, True, False, False], [False, True, False, False]])
    losses = study.policy_losses(logits, targets, mating, 'set', 1)
    torch.testing.assert_close(losses, torch.tensor([-math.log(.5), -math.log(.6)], dtype=torch.float64))
    single = study.policy_losses(logits, targets, mating, 'single', 1)
    assert losses[0] < single[0] and losses[1] == single[1]


def test_single_and_set_losses_and_gradients_agree_when_each_target_set_is_singleton(study):
    base = torch.tensor([[1., -2., .5], [-1., 3., 2.], [0., .2, -.3]])
    targets = torch.tensor([0, 2, 1])
    mating = F.one_hot(targets, num_classes=3).bool()
    gradients, losses = [], []
    for arm in ('single', 'set'):
        logits = base.clone().requires_grad_(True)
        loss = study.policy_losses(logits, targets, mating, arm, 2)
        loss.sum().backward()
        gradients.append(logits.grad)
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_mate_loss_is_invariant_to_recorded_winner_but_single_loss_is_not(study):
    logits = torch.tensor([[0., 2., 1.]])
    mating = torch.tensor([[True, True, False]])
    first = study.policy_losses(logits, torch.tensor([0]), mating, 'set', 1)
    other = study.policy_losses(logits, torch.tensor([1]), mating, 'set', 1)
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    assert not torch.equal(study.policy_losses(logits, torch.tensor([0]), mating, 'single', 1),
                           study.policy_losses(logits, torch.tensor([1]), mating, 'single', 1))


def test_masked_loss_gradients_are_finite_and_illegal_actions_receive_zero_gradient(study):
    raw = torch.tensor([[.5, -.2, 1000.], [1., -1., 1000.]], requires_grad=True)
    legal = torch.tensor([[True, True, False], [True, True, False]])
    mating = torch.tensor([[True, True, False], [False, True, False]])
    logits = raw.masked_fill(~legal, -torch.inf)
    loss = study.policy_losses(logits, torch.tensor([0, 1]), mating, 'set', 1).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad[~legal]) == 0
    assert raw.grad[1, :2].abs().sum() > 0
    invalid = mating.clone()
    invalid[0, 2] = True
    with pytest.raises(ValueError, match='legal'):
        study.policy_losses(logits, torch.tensor([0, 1]), invalid, 'set', 1)


@pytest.mark.parametrize('case', ['empty', 'dtype', 'shape', 'arm', 'count', 'wrong_recorded'])
def test_invalid_set_targets_are_rejected(study, case):
    logits = torch.tensor([[1., 2., 3.]])
    targets, mating, arm, count = torch.tensor([0]), torch.tensor([[True, False, False]]), 'set', 1
    if case == 'empty':
        mating.zero_()
    elif case == 'dtype':
        mating = mating.float()
    elif case == 'shape':
        mating = mating[:, :2]
    elif case == 'arm':
        arm = 'other'
    elif case == 'count':
        count = 2
    else:
        targets[0] = 1
    with pytest.raises(ValueError):
        study.policy_losses(logits, targets, mating, arm, count)


def test_label_changes_never_change_model_inputs(study):
    original = mate_row()
    altered = {**original, 'target_uci': original['mating_uci'][-1],
               'mating_uci': [original['mating_uci'][-1]], 'target_value': -.7,
               'engine_score': -999, 'human_comment': 'not a model input'}
    first, first_names = study.tensors([original])
    second, second_names = study.tensors([altered])
    assert first_names == second_names
    for key in ('observations', 'candidates', 'mask'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in ('targets', 'mating', 'values'):
        assert not torch.equal(first[key], second[key])


def test_schedule_is_exact_paired_unique_replay_and_complete_mate_epochs(study):
    random.seed(1907)
    previous = random.getstate()
    single = study.batch_schedule(17, 32768)
    paired = study.batch_schedule(17, 32768)
    assert random.getstate() == previous
    assert single == paired and single != study.batch_schedule(29, 32768)
    assert len(single) == study.PROTOCOL['updates_per_fit'] == 192
    replay = [index for batch in single for index in batch['replay']]
    assert len(replay) == len(set(replay)) == 12288
    assert min(replay) >= 0 and max(replay) < 32768
    assert all(len(batch['mate']) == len(batch['replay']) == 64 for batch in single)
    for epoch in range(1, 13):
        mates = [index for batch in single if batch['epoch'] == epoch for index in batch['mate']]
        assert sorted(mates) == list(range(1024))
    with pytest.raises(ValueError, match='Not enough replay'):
        study.batch_schedule(17, 12287)


def test_combined_batch_pads_candidate_axes_and_preserves_labels_and_order(study):
    mates, _ = study.tensors([mate_row('m0'), mate_row('m1')])
    replay, _ = study.tensors([row('r0'), row('r1'), row('r2')])
    assert mates['mask'].shape[1] != replay['mask'].shape[1]
    combined = study.combined_batch(mates, replay, {'mate': [1, 0], 'replay': [2, 0]}, 'cpu')
    width = max(mates['mask'].shape[1], replay['mask'].shape[1])
    assert combined['candidates'].shape == (4, width, 5)
    assert combined['mask'].dtype == combined['mating'].dtype == torch.bool
    for offset, source, indices in [(0, mates, [1, 0]), (2, replay, [2, 0])]:
        for index, source_index in enumerate(indices):
            out = offset+index
            count = source['mask'].shape[1]
            for key in ('mask', 'mating', 'candidates'):
                torch.testing.assert_close(combined[key][out, :count], source[key][source_index])
                assert not combined[key][out, count:].any()
            for key in ('targets', 'values', 'observations'):
                torch.testing.assert_close(combined[key][out], source[key][source_index])
    assert torch.all(combined['mating'] <= combined['mask'])


class SyntheticPolicy(nn.Module):
    """One scalar test double, not SpatialChess weights or real model inference."""

    def __init__(self, seed=17):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.02))
        self.mode, self.seed, self.width, self.depth = 'predict', seed, 32, 4

    def forward(self, observations, candidates, mask, depth=4):
        assert depth == 4 and observations.shape[1:] == (19, 8, 8)
        logits = self.weight*candidates[:, :, 1].float()/63
        return logits.masked_fill(~mask, -torch.inf), self.weight.tanh().expand(len(observations)), observations

    def save(self, path, *, plan_sha256):
        with Path(path).open('xb') as stream:
            torch.save({'seed': self.seed, 'weight': self.weight.detach().clone(),
                        'plan_sha256': plan_sha256}, stream)

    @classmethod
    def load(cls, path, *, expected_plan_sha256):
        payload = torch.load(path, weights_only=True)
        if payload['plan_sha256'] != expected_plan_sha256:
            raise ValueError('Synthetic checkpoint plan mismatch')
        model = cls(payload['seed'])
        with torch.no_grad():
            model.weight.copy_(payload['weight'])
        return model


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.fixture
def tiny_pipeline(study, tmp_path, monkeypatch):
    config = {**copy.deepcopy(study.PROTOCOL), 'seeds': [17], 'mate_sizes': [4, 2, 2],
              'epochs': 2, 'batch_size': 4, 'mate_per_batch': 2, 'updates_per_fit': 4,
              'replay_count_per_fit': 8, 'device': 'cpu'}
    monkeypatch.setattr(study, 'PROTOCOL', config)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'SpatialChess', SyntheticPolicy)
    monkeypatch.setattr(study, 'EXCLUSIONS', [f'{study.DATA}/train.jsonl'])
    for split, count in [('train', 8), ('dev', 2), ('shift', 2)]:
        write_rows(tmp_path/study.DATA/f'{split}.jsonl', [row(f'{split}-{index}') for index in range(count)])
    old = tmp_path/study.OLD_PLAN
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({'puzzles': {'positions': [{'source_game': 'https://lichess.org/OldGame1', 'solver_fen': chess.STARTING_FEN}]}}))
    spatial = tmp_path/study.SPATIAL_PLAN
    spatial.parent.mkdir(parents=True, exist_ok=True)
    spatial.write_text('{}\n')
    csv = tmp_path/study.CSV
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text('Synthetic CSV marker; selector is a test double.\n')
    weights = tmp_path/'models/chess-spatial-v1/predict-17/weights.pt'
    weights.parent.mkdir(parents=True)
    SyntheticPolicy().save(weights, plan_sha256=study.sha(spatial))
    selected = {'splits': {split: [mate_row(f'{split}-{i}') for i in range(count)]
                           for split, count in zip(('train', 'dev', 'confirm'), (4, 2, 2), strict=True)},
                'selection': {'scope': 'Synthetic fixture only'}}

    def select(path, games, excluded, *, sizes, seed):
        assert path == csv and games == {'https://lichess.org/OldGame1'}
        assert len(excluded) == 1 and sizes == (4, 2, 2)
        assert seed == config['selection_seed']
        return copy.deepcopy(selected)

    monkeypatch.setattr(study, 'select_mates', select)
    monkeypatch.setattr(study, 'signature', lambda: {'protocol': copy.deepcopy(config),
                                                   'inputs': {'synthetic.csv': study.sha(csv)}})
    plan = study.prepare(tmp_path/'prepared')
    return study, plan, tmp_path/'execution', tmp_path/'summary.json', selected


def test_synthetic_pipeline_has_paired_initialization_exact_updates_and_reproducible_report(
    tiny_pipeline, monkeypatch
):
    study, plan, execution, summary_path, selected = tiny_pipeline
    loader, loaded = study.SpatialChess.load, []

    def track_initialization(cls, path, *, expected_plan_sha256):
        model = loader(path, expected_plan_sha256=expected_plan_sha256)
        loaded.append((Path(path), model.weight.detach().clone()))
        return model

    monkeypatch.setattr(study.SpatialChess, 'load', classmethod(track_initialization))
    _, verified_selection = study.verify_plan(plan)
    assert verified_selection == selected
    assert not (execution/'started.json').exists()
    study.run(plan, execution)
    assert len(loaded) == 3 and len({path for path, _ in loaded}) == 1
    assert all(torch.equal(weight, loaded[0][1]) for _, weight in loaded)
    summary = study.report(plan, execution, summary_path)
    assert summary['updates'] == 8 and summary['new_training_engine_calls'] == 0
    assert set(summary['results']) == {'frozen-17', 'single-17', 'set-17'}
    for name, result in summary['results'].items():
        assert result['updates'] == (0 if name == 'frozen-17' else 4)
        arm = name.split('-')[0]
        for split, metric in result['metrics'].items():
            assert summary['mean_accuracy'][arm][split] == metric['accuracy']
    for arm in ('single', 'set'):
        logs = study.read_rows(execution/f'{arm}-17/learning.jsonl')
        assert [item['epoch'] for item in logs] == [1, 1, 2, 2]
        for item in logs:
            assert item['loss'] == pytest.approx(.5*(item['mate_policy_loss']+item['replay_policy_loss'])
                                                +.5*item['value_loss'], abs=2e-6)
    assert (execution/'single-17/weights.pt').exists() and (execution/'set-17/weights.pt').exists()
    assert not (execution/'frozen-17/weights.pt').exists()
    with pytest.raises(FileExistsError):
        study.run(plan, execution)


def rebind_fit(study, execution, name, filename):
    fit = execution/name
    receipt = json.loads((fit/'completed.json').read_text())
    receipt['files'][filename] = study.sha(fit/filename)
    (fit/'completed.json').write_text(json.dumps(receipt)+'\n')
    complete = json.loads((execution/'completed.json').read_text())
    complete['fits'][name] = study.sha(fit/'completed.json')
    (execution/'completed.json').write_text(json.dumps(complete)+'\n')


@pytest.mark.parametrize('change', ['correctness', 'training_loss', 'epoch', 'schedule', 'updates', 'missing_fit'])
def test_rehashed_mutations_cannot_fabricate_a_completed_report(tiny_pipeline, change):
    study, plan, execution, output, _ = tiny_pipeline
    study.run(plan, execution)
    if change == 'correctness':
        path = execution/'set-17/mate_confirm.jsonl'
        rows = study.read_rows(path)
        rows[0]['correct'] = not rows[0]['correct']
        write_rows(path, rows)
        rebind_fit(study, execution, 'set-17', path.name)
    elif change in ('training_loss', 'epoch', 'updates'):
        path = execution/'single-17/learning.jsonl'
        rows = study.read_rows(path)
        if change == 'training_loss':
            rows[0]['loss'] += 1
        elif change == 'epoch':
            rows[0]['epoch'] = 12
        else:
            rows.pop()
        write_rows(path, rows)
        rebind_fit(study, execution, 'single-17', path.name)
    elif change == 'schedule':
        path = execution/'schedule.json'
        schedule = json.loads(path.read_text())
        schedule['batches']['17'][0]['replay'][0] = schedule['batches']['17'][0]['replay'][1]
        path.write_text(json.dumps(schedule)+'\n')
        complete = json.loads((execution/'completed.json').read_text())
        complete['schedule_sha256'] = study.sha(path)
        (execution/'completed.json').write_text(json.dumps(complete)+'\n')
    else:
        complete = json.loads((execution/'completed.json').read_text())
        complete['fits'].pop('set-17')
        (execution/'completed.json').write_text(json.dumps(complete)+'\n')
    with pytest.raises(ValueError):
        study.report(plan, execution, output)
    assert not output.exists()


def test_plan_rejects_changed_selected_positions(tiny_pipeline):
    study, plan, _, _, _ = tiny_pipeline
    selection = plan.parent/'selection.json'
    selected = json.loads(selection.read_text())
    selected['splits']['confirm'][0]['target_uci'] = 'a1a2'
    selection.write_text(json.dumps(selected)+'\n')
    with pytest.raises(ValueError, match='Frozen'):
        study.verify_plan(plan)


def test_runtime_failure_leaves_receipt_without_retry(tiny_pipeline, monkeypatch):
    study, plan, execution, _, _ = tiny_pipeline
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('Synthetic failure')

    monkeypatch.setattr(study, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='Synthetic failure'):
        study.run(plan, execution)
    assert calls == [1]
    assert json.loads((execution/'failed.json').read_text())['error_type'] == 'RuntimeError'
    assert not (execution/'completed.json').exists()
