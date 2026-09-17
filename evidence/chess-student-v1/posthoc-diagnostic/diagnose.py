"""Post-hoc, no-fitting inspection of all final chess-student-v1 checkpoints."""
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from openjev.research.chess_student import ChessStudent, MODES, UCI_MOVES
from openjev.research.chess_arena import greedy_material_policy

spec = importlib.util.spec_from_file_location('frozen_train', ROOT/'scripts/train_chess_student.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
OUT = Path(__file__).resolve().parent
RUN = ROOT/'runs/chess-student-v1/execution'
PLAN = ROOT/'evidence/chess-student-v1/plan.json'


def read(path):
    return json.loads(path.read_text())


def write(name, obj):
    with (OUT/name).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def forward(model, data, permutation=None):
    choices, probs, vals, hidden_all, encoded_all = [], [], [], [], []
    for start in range(0, len(data['targets']), 64):
        ids = torch.arange(start, min(start+64, len(data['targets'])))
        obs = data['observations'][ids if permutation is None else permutation[ids]]
        encoded = model.encoder(obs)
        hidden = torch.zeros_like(encoded)
        for _ in range(4):
            hidden = model.core(encoded, hidden)
        logits = model.policy_head(hidden).masked_fill(~data['mask'][ids], -torch.inf)
        probability = logits.softmax(-1)
        choices.extend(probability.argmax(-1).tolist())
        probs.extend(probability[torch.arange(len(ids)), data['targets'][ids]].tolist())
        vals.extend(model.value_head(hidden).tanh().squeeze(-1).tolist())
        hidden_all.append(hidden)
        encoded_all.append(encoded)
    choices, vals, probs = np.array(choices), np.array(vals), np.array(probs)
    target = data['targets'].numpy()
    values = data['values'].numpy()
    hidden = torch.cat(hidden_all)
    encoded = torch.cat(encoded_all)
    metrics = {
        'agreement': float(np.mean(choices == target)),
        'nll': float(np.mean(-np.log(np.maximum(probs, 1e-12)))),
        'value_mae': float(np.mean(np.abs(vals-values))),
        'value_mse': float(np.mean((vals-values)**2)),
        'value_std': float(np.std(vals)),
        'value_correlation': float(np.corrcoef(vals, values)[0, 1]),
        'hidden_std_across_positions_mean': float(hidden.std(0).mean()),
        'encoded_std_across_positions_mean': float(encoded.std(0).mean()),
        'hidden_saturation_abs_gt_095_fraction': float((hidden.abs() > .95).float().mean()),
        'encoder_saturation_abs_gt_095_fraction': float((encoded.abs() > .95).float().mean()),
    }
    return metrics, choices, probs, vals


def categories(rows, frequencies):
    output = []
    for row in rows:
        board, move = chess.Board(row['fen']), chess.Move.from_uci(row['target_uci'])
        frequency = frequencies[move.uci()]
        output.append({
            'target_piece': chess.piece_name(board.piece_type_at(move.from_square)),
            'target_capture': str(board.is_capture(move)),
            'target_check': str(board.gives_check(move)),
            'in_check': str(board.is_check()),
            'promotion': str(bool(move.promotion)),
            'castle': str(board.is_castling(move)),
            'target_frequency_bin': '0' if frequency == 0 else '1-5' if frequency <= 5 else '6-20' if frequency <= 20 else '21+',
            'ply_bin': '0-9' if row['ply'] < 10 else '10-19' if row['ply'] < 20 else '20-29' if row['ply'] < 30 else '30-39',
        })
    return output


def grouped(rows, predictions, cats):
    result = {}
    correct = np.array([move == row['target_uci'] for move, row in zip(predictions, rows)])
    for key in cats[0]:
        result[key] = {}
        for value in sorted({cat[key] for cat in cats}):
            ix = np.array([cat[key] == value for cat in cats])
            result[key][value] = {'count': int(ix.sum()), 'agreement': float(correct[ix].mean())}
    return result


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan, complete = read(PLAN), read(RUN/'completed.json')
    for source, expected in plan['sources'].items():
        assert train.sha(ROOT/source) == expected, source
    assert complete['plan_sha256'] == train.sha(PLAN)
    assert complete['status'] == 'completed' and complete['fits'] == 9
    rows = train.validate_data(RUN/'data')
    data = {split: train.tensors(values) for split, values in rows.items()}
    write('scope.json', {
        'scope': 'Post-hoc diagnosis of existing v1 train/development data; no new holdout, fitting, tuning, teacher calls, or efficacy claim',
        'plan_sha256': train.sha(PLAN), 'source_sha256': train.sha(Path(__file__)),
        'evaluations': 'All 9 final weights: 4096 training +1024 development +fixed circularly permuted development features; backward only on first64 original training examples; no optimizer',
        'permutation': 'torch.roll(torch.arange(1024), shifts=137); retain original legal mask, targets and labels',
        'feature_permutation_scope': 'Post-hoc sensitivity ablation, invalid feature/mask combinations; legal masks retain position information, not an unbiased efficacy estimate',
        'heuristic_scope': 'Existing deterministic one-ply greedy material, scored on the already inspected development set',
    })
    frequencies = Counter(row['target_uci'] for row in rows['train'])
    cats = categories(rows['dev'], frequencies)
    baseline = train.simple_baselines(rows['train'], rows['dev'])
    majority = [p['choice'] for p in baseline['training_move_frequency']['predictions']]
    greedy = greedy_material_policy()
    greedy_choices = [greedy(chess.Board(row['fen']))['choice'] for row in rows['dev']]
    legal_counts = data['dev']['mask'].sum(-1).numpy()
    target_values = data['dev']['values'].numpy()
    dataset = {
        'train_games': len({row['game_id'] for row in rows['train']}),
        'dev_games': len({row['game_id'] for row in rows['dev']}),
        'train_distinct_target_uci': len(frequencies),
        'dev_distinct_target_uci': len({row['target_uci'] for row in rows['dev']}),
        'dev_target_unseen_in_train_count': sum(frequencies[row['target_uci']] == 0 for row in rows['dev']),
        'top10_train_targets': frequencies.most_common(10),
        'train_top20_target_fraction': sum(n for _, n in frequencies.most_common(20))/4096,
        'dev_legal_count_mean': float(legal_counts.mean()),
        'dev_values_mean': float(target_values.mean()), 'dev_values_std': float(target_values.std()),
        'constant_train_mean_value_mae': float(np.mean(np.abs(target_values-data['train']['values'].numpy().mean()))),
        'constant_train_mean_value_mse': float(np.mean((target_values-data['train']['values'].numpy().mean())**2)),
        'uniform_expected_agreement': float(np.mean(1/legal_counts)),
        'majority_agreement': float(np.mean([a == r['target_uci'] for a, r in zip(majority, rows['dev'])])),
        'greedy_material_posthoc_agreement': float(np.mean([a == r['target_uci'] for a, r in zip(greedy_choices, rows['dev'])])),
        'majority_groups': grouped(rows['dev'], majority, cats),
        'greedy_material_groups': grouped(rows['dev'], greedy_choices, cats),
    }
    output = []
    for mode in MODES:
        for seed in (17, 29, 43):
            name = f'{mode}-{seed}'
            path = RUN/name
            receipt = read(path/'completed.json')
            assert train.sha(path/'completed.json') == complete['fit_receipts'][name]
            for file, expected in receipt['files'].items():
                assert train.sha(path/file) == expected, (name, file)
            logs = train.jsonl(path/'learning.jsonl')
            assert len(logs) == receipt['updates'] == 192
            assert [row['update'] for row in logs] == list(range(1, 193))
            assert logs[-1]['examples_seen'] == receipt['examples_seen'] == 12288
            model = ChessStudent.load(path/'weights.pt', expected_plan_sha256=train.sha(PLAN))
            original = digest(model)
            train_metrics, _, _, _ = forward(model, data['train'])
            dev_metrics, choices, probs, vals = forward(model, data['dev'])
            saved = read(path/'evaluation.json')['predictions']
            assert [UCI_MOVES[c] for c in choices] == [p['choice'] for p in saved]
            assert np.max(np.abs(probs-np.array([p['target_probability'] for p in saved]))) < 1e-7
            assert np.max(np.abs(vals-np.array([p['value'] for p in saved]))) < 1e-7
            perm_metrics, perm_choices, _, _ = forward(model, data['dev'], torch.roll(torch.arange(1024), 137))
            initial = ChessStudent(mode, seed)
            delta = (model.encoder[0].weight-initial.encoder[0].weight).detach().norm(dim=1)
            model.zero_grad(set_to_none=True)
            logits, values = model(data['train']['observations'][:64], data['train']['mask'][:64])
            loss = F.cross_entropy(logits, data['train']['targets'][:64]) + .5*F.mse_loss(values, data['train']['values'][:64])
            loss.backward()
            grad = model.encoder[0].weight.grad.norm(dim=1)
            paths = {'encoder_changed_rows': int((delta > 0).sum()),
                     'encoder_nonzero_gradient_rows': int((grad > 0).sum()),
                     'encoder_gradient_row_l2': grad.tolist(),
                     'encoder_weight_change_row_l2': delta.tolist(),
                     'encoder_effectively_disconnected_parameters': int((grad == 0).sum())*842,
                     'parameter_count': model.parameter_count(),
                     'core_parameters': sum(p.numel() for p in model.core.parameters()),
                     'core_nonzero_gradient_parameters': sum(int((p.grad != 0).sum()) for p in model.core.parameters()),
                     'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
                     'policy_head_parameters': sum(p.numel() for p in model.policy_head.parameters())}
            if mode != 'gru':
                reachable = model.core.drive_mask.bool().clone()
                counts = [int(reachable.sum())]
                for _ in range(3):
                    reached = reachable.clone()
                    reached[model.core.target[reachable[model.core.source]]] = True
                    reachable = reached
                    counts.append(int(reachable.sum()))
                paths['input_reachable_nodes_by_update'] = counts
            assert digest(model) == original
            assert train.sha(path/'weights.pt') == receipt['files']['weights.pt']
            epoch_metrics = [{'epoch': ep, **{key: float(np.mean([r[key] for r in logs if r['epoch'] == ep])) for key in ('policy_ce', 'value_mse', 'loss', 'gradient_norm')}} for ep in (1, 2, 3)]
            output.append({
                'mode': mode, 'seed': seed, 'checkpoint_sha256': receipt['files']['weights.pt'],
                'learning_epochs': epoch_metrics,
                'first10_ce': float(np.mean([r['policy_ce'] for r in logs[:10]])),
                'last10_ce': float(np.mean([r['policy_ce'] for r in logs[-10:]])),
                'gradient_clip_fraction': float(np.mean([r['gradient_norm'] > 1 for r in logs])),
                'training_wall_seconds': receipt['training_wall_seconds'],
                'train': train_metrics, 'dev': dev_metrics,
                'permuted_dev_features': perm_metrics,
                'permuted_features_choice_unchanged_fraction': float(np.mean(choices == perm_choices)),
                'majority_choice_agreement_fraction': float(np.mean([UCI_MOVES[c] == p for c, p in zip(choices, majority)])),
                'groups': grouped(rows['dev'], [UCI_MOVES[c] for c in choices], cats),
                'gradient_paths': paths,
            })
    summary = {'scope_sha256': train.sha(OUT/'scope.json'), 'dataset': dataset, 'fits': output,
               'checks': {'all9_checkpoint_and_receipt_hashes_match': True,
                          'all9_saved_dev_predictions_reproduced': True,
                          'all9_checkpoints_unchanged': True,
                          'all9_logs_192_updates_12288_examples': True,
                          'source_hashes_match_frozen_plan': True},
               'source_sha256': train.sha(Path(__file__))}
    write('summary.json', summary)
    print(json.dumps({'dataset': {k: v for k, v in dataset.items() if not k.endswith('groups')},
                      'fits': [{k: v for k, v in f.items() if k not in ('groups', 'gradient_paths')} for f in output]}, indent=2))


if __name__ == '__main__':
    main()
