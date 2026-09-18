import sys
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import chess_graph_contrast_study as study
from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_child_graph_cache import write_cache


def metrics():
    return {f'{arm}-{seed}': {split: {'agreement': .34 if arm == 'contrast' else .30}
                             for split in ('dev', 'shift')}
            for arm in ('contrast', 'base', 'root', 'child', 'permuted_contrast')
            for seed in (97, 109, 127)}


@pytest.mark.parametrize('split', ('dev', 'shift'))
@pytest.mark.parametrize('comparator', ('base', 'root', 'child', 'permuted_contrast'))
def test_each_comparison_on_each_panel_is_required(split, comparator):
    values = metrics(); assert len(study.gate(values)) == 8
    assert all(c['passed'] for c in study.gate(values))
    # A positive half-point effect is too small for an ingredient comparison.
    changed = .35 if comparator == 'base' else .335
    for seed in (97, 109, 127): values[f'{comparator}-{seed}'][split]['agreement'] = changed
    checks = study.gate(values)
    assert sum(not c['passed'] for c in checks) == 1
    failed = next(c for c in checks if not c['passed'])
    assert (failed['split'], failed['comparator']) == (split, comparator)


def test_paired_seed_loss_cannot_be_hidden_by_mean_gain():
    values = metrics(); values['contrast-97']['dev']['agreement'] = .29
    values['contrast-109']['dev']['agreement'] = .42
    checks = [c for c in study.gate(values) if c['split'] == 'dev']
    assert all(c['mean_gain'] > .01 and not c['passed'] for c in checks)


def test_budget_and_seed_matched_controls_are_fixed():
    p = study.PROTOCOL; order = study.fit_order()
    assert order == study.fit_order() and set(order) == {'97', '109', '127'}
    assert all(sorted(v) == sorted(study.ARMS) for v in order.values())
    assert p['fits'] == 12 and p['total_updates'] == 18432
    assert p['epochs'] * p['training_examples'] // p['batch_size'] == p['updates_per_fit'] == 1536
    assert p['head_parameters'] == 16744 and p['time_cap_seconds'] == 7200


def test_cached_batch_and_complete_native_decisions_agree(tmp_path):
    torch.set_num_threads(2)
    fens = [chess.STARTING_FEN, '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2',
            '4k3/P7/8/8/8/8/8/4K3 w - - 0 1']
    fens += [chess.Board(f).mirror().fen(en_passant='fen') for f in fens]
    boards = [chess.Board(f) for f in fens]; inputs, menus = encode_batch(boards, 'direct')
    model = CandidateChess('direct', seed=31, width=32).eval().requires_grad_(False)
    write_cache([{'fen': f} for f in fens], tmp_path/'cache', provenance={'test': True})
    graphs = study.ChildGraphCache(tmp_path/'cache')
    data = {'observations': inputs['observations'], 'candidates': inputs['candidates'],
            'mask': inputs['legal_mask'], 'targets': torch.zeros(len(boards), dtype=torch.long),
            'edges': study.candidate_graphs(boards)['root']}
    features = study.prior.root_cache(model, data)
    index = torch.tensor([4, 1, 3, 1, 2, 0, 5]); graph = graphs.batch(index)
    args = study.arguments(model, data, features, index, graph)
    assert args[-1].shape[0] == int(args[3].sum())
    for arm in study.ARMS:
        head = study.SharedRootGraphContrastHead(arm, seed=1131)
        with torch.no_grad():
            head.output.weight.fill_(.1)
            logits = head(*args, permutation=graph['permutation'])
        for j, i in enumerate(index.tolist()):
            assert graph['menus'][j] == menus[i]
            expected = menus[i][int(logits[j].argmax())]
            assert study.decision(model, head, fens[i]) == expected
    bad = dict(graph, root=graph['root'].clone()); bad['root'][0, 0, 0, 0] = 1
    with pytest.raises(AssertionError): study.arguments(model, data, features, index, bad)


def test_corruption_diagnostic_uses_fitted_contrast_checkpoint(tmp_path):
    directory = tmp_path/'contrast-97'; directory.mkdir()
    head = study.SharedRootGraphContrastHead('contrast', seed=1197)
    with torch.no_grad(): head.output.weight.fill_(.125)
    study.prior.save(directory/'weights.pt', {'version': study.VERSION, 'arm': 'contrast',
        'seed': 97, 'plan_sha256': 'test', 'state_dict': head.state_dict()})
    corrupted = study.load(tmp_path, 97, 'contrast_corrupted', 'test')
    assert corrupted.contrast_arm == 'permuted_contrast'
    assert all(torch.equal(v, corrupted.state_dict()[k]) for k, v in head.state_dict().items())
    with pytest.raises(AssertionError): study.load(tmp_path, 97, 'contrast_corrupted', 'other-plan')
