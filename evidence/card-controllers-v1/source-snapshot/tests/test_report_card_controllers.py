"""Fake saved arrays and receipts only; no environment, model, or RNG calls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openjev.research.card_memory_picker import CardPolicyTracker


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[1] / 'scripts/report_card_controllers.py'
    spec = importlib.util.spec_from_file_location('report_card_controllers_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_rows(report):
    result = {}
    for policy, value in [('A', .05), ('B', .1), ('C', .13)]:
        result[policy] = {}
        for name in report.NAMES:
            val = .8 if name in report.REFERENCES else value
            result[policy][name] = {'per_seed_returns': [val] * 64, 'mean_return': val,
                                    'per_seed_successes': [name in report.REFERENCES] * 64,
                                    'wall_seconds': 1., 'native_steps': 64 * 104}
    return result


def change(rows, policy, name, value):
    rows[policy][name]['mean_return'] = value
    rows[policy][name]['per_seed_returns'] = [value] * 64


def test_exact_inclusive_decimal_margin_and_all_coverage(report):
    families, contrasts, gate = report.aggregate(fake_rows(report))
    assert gate['passed'] and gate['checks_passed'] == gate['total_checks'] == 13
    assert contrasts['C_minus_B']['learned_mean_difference'] == .03
    assert families['C']['delta']['episodes'] == 192
    assert families['C']['exact']['episodes'] == 64
    assert len(contrasts['B_minus_A']['per_fit']) == 20
    assert len(contrasts['C_minus_B']['per_fit']['gru-pair2']['paired_deck_return_differences']) == 64


def test_below_margin_cannot_round_up(report):
    rows = fake_rows(report)
    for name in report.NAMES[:-2]:
        change(rows, 'C', name, .12999999999)
    _, _, gate = report.aggregate(rows)
    assert not gate['passed'] and not gate['checks'][0]['passed']


def test_zero_pair_is_not_positive_and_other_families_cannot_rescue(report):
    rows = fake_rows(report)
    change(rows, 'C', 'gru-pair0', .1)
    change(rows, 'C', 'gru-pair1', .1)
    change(rows, 'C', 'gru-pair2', .5)
    _, _, gate = report.aggregate(rows)
    checks = {r['name']: r for r in gate['checks']}
    assert checks['learned_mean_C_minus_B_at_least_0.03']['passed']
    assert not checks['gru_at_least_two_positive_fits']['passed']
    assert not gate['passed']


def test_two_positive_fits_do_not_rescue_negative_family_mean(report):
    rows = fake_rows(report)
    change(rows, 'C', 'gru-pair0', -.1)
    _, _, gate = report.aggregate(rows)
    checks = {r['name']: r for r in gate['checks']}
    assert checks['gru_at_least_two_positive_fits']['passed']
    assert not checks['gru_mean_nonnegative']['passed']


@pytest.mark.parametrize('kind', ['missing_family', 'missing_policy', 'missing_deck', 'nan'])
def test_incomplete_or_nonfinite_never_dropped(report, kind):
    rows = fake_rows(report)
    if kind == 'missing_family':
        del rows['A']['delta-pair2']
    elif kind == 'missing_policy':
        del rows['C']
    elif kind == 'missing_deck':
        rows['B']['delta-pair2']['per_seed_returns'].pop()
    else:
        rows['C']['delta-pair2']['per_seed_returns'][0] = float('nan')
    with pytest.raises(ValueError):
        report.aggregate(rows)


def test_exact_3840_episode_members_and_no_extra_files(report):
    members = report.expected_members()
    assert len(members) == 7742
    assert sum(n.endswith('.npz') for n in members) == 3840
    assert sum(n.endswith('/completed.json') for n in members) == 60
    assert 'controllers/gru-pair2/C/episodes/063.json' in members


def episode(report, tmp_path, policy):
    tracker = CardPolicyTracker(policy)
    frame = np.full(52, 13, dtype=np.int64)
    tracker.reset(frame)
    raw = np.full((52, 13), 1/13, dtype=np.float64)
    arrays = {k: [] for k in ('observations', 'actions', 'ranks', 'rewards', 'terminated', 'truncated',
                              'raw_probabilities', 'picker_probabilities')}
    arrays['observations'].append(frame.copy())
    diagnostics = []
    for step in range(52):
        action, probs, info = tracker.decision(raw)
        assert action == step
        after = frame.copy()
        after[action] = action // 4
        reward = 0.0 if step % 2 == 0 else 2/52
        event = tracker.observe(action, reward, after)
        for key, value in [('actions', action), ('ranks', event.rank), ('rewards', reward),
                           ('terminated', step == 51), ('truncated', False), ('raw_probabilities', raw.copy()),
                           ('picker_probabilities', probs), ('observations', after.copy())]:
            arrays[key].append(value)
        diagnostics.append(info)
        frame = after
    arrays = {k: np.array(v) for k, v in arrays.items()}
    path = tmp_path / 'episode.npz'
    np.savez(path, **arrays)
    receipt = {'status': 'complete', 'controller': 'gated_delta-pair0', 'policy': policy,
               'npz_sha256': report.sha(path), 'npz_bytes': path.stat().st_size, 'native_steps': 52, 'array_bytes': sum(a.nbytes for a in arrays.values()),
               'decisions': diagnostics, 'return': 1., 'success': True, 'matched_pairs': 26,
               'counts': {k: (1 if 'reset' in k or 'identity' in k or 'init' in k else 52) for k in (
                   'reset_attempted', 'reset_returned', 'identity_read_attempted', 'identity_read_returned',
                   'init_attempted', 'init_returned', 'decision_attempted', 'decision_returned',
                   'native_attempted', 'native_returned', 'predict_attempted', 'predict_returned',
                   'write_attempted', 'write_returned')}, 'timings': {'native_seconds': .1}, 'wall_seconds': .2}
    return path, receipt, arrays


@pytest.mark.parametrize('policy', ['A', 'B', 'C'])
def test_public_trace_and_all_picker_diagnostics(report, tmp_path, policy):
    path, receipt, _ = episode(report, tmp_path, policy)
    result = report.audit_episode(path, receipt, policy=policy, controller=receipt['controller'])
    assert result['success'] and result['unique_positions'] == 52
    np.testing.assert_array_equal(result['layout'], np.arange(52) // 4)


@pytest.mark.parametrize('corruption', ['diagnostic', 'probabilities', 'action', 'native_count', 'identity_count'])
def test_resealed_corruption_rejected(report, tmp_path, corruption):
    path, receipt, arrays = episode(report, tmp_path, 'C')
    if corruption == 'diagnostic':
        receipt['decisions'][0]['tie_size'] += 1
    elif corruption == 'native_count':
        receipt['counts']['native_returned'] -= 1
    elif corruption == 'identity_count':
        receipt['counts']['identity_read_returned'] = 0
    else:
        if corruption == 'probabilities':
            arrays['picker_probabilities'][0, 0, 0] = .9
        else:
            arrays['actions'][0] = 1
        np.savez(path, **arrays)
        receipt['npz_sha256'], receipt['npz_bytes'] = report.sha(path), path.stat().st_size
    with pytest.raises(ValueError):
        report.audit_episode(path, receipt, policy='C', controller=receipt['controller'])


def test_public_layout_hash_and_conflicting_reveal(report):
    layout = np.full(52, -1, dtype=np.int64)
    revealed = np.arange(52, dtype=np.int64) // 4
    partial = revealed.copy(); partial[26:] = -1
    report.merge_public_layout(layout, partial)
    with pytest.raises(ValueError, match='Incomplete'):
        report.layout_digest(layout)
    report.merge_public_layout(layout, revealed)
    assert len(report.layout_digest(layout)) == 64
    changed = revealed.copy(); changed[0] = 5
    with pytest.raises(ValueError, match='conflicting'):
        report.merge_public_layout(layout, changed)


@pytest.mark.parametrize('corrupt_counter', [False, True])
def test_outer_exact_members_and_terminal_receipts(report, tmp_path, monkeypatch, corrupt_counter):
    """Outer wiring with actual 7,742 fake files; numerical episode replay is separately tested."""
    root = tmp_path; evaluation = root/'execution'; evaluation.mkdir()
    source_files = {key: root/(key+'.json') for key in ('protocol', 'inputs', 'bindings', 'checkpoint_map')}
    for path in source_files.values():
        path.write_text('{}')
    expected = {key: report.sha(path) for key, path in source_files.items()}
    recipe = {'evaluation': {'output_cap_bytes': 6_000_000_000, 'wall_cap_seconds': 1800}, 'lineage': {}}
    cases = {'evaluation': [{'seed': i, 'policy_order': list(report.POLICIES[i%3:]+report.POLICIES[:i%3])} for i in range(64)]}
    maps, restored = {}, {}
    for name in report.NAMES[:-2]:
        fit = root/(name+'.json'); report.write(fit, {'final_weights_sha256': 'a'*64})
        maps[name] = {'checkpoint_sha256': 'b'*64, 'completed_path': fit.name}
        restored[name] = {'checkpoint_sha256': 'b'*64, 'weights_sha256': 'a'*64}
    monkeypatch.setattr(report, 'authenticate_inputs', lambda *_: (recipe, cases, maps, {'files': {}}))
    layout = np.arange(52, dtype=np.int64)//4; digest = report.layout_digest(layout)
    work_names = ('reset_attempted', 'reset_returned', 'identity_read_attempted', 'identity_read_returned',
                  'init_attempted', 'init_returned', 'predict_attempted', 'predict_returned',
                  'decision_attempted', 'decision_returned', 'native_attempted', 'native_returned',
                  'write_attempted', 'write_returned')
    totals = dict.fromkeys(work_names, 0)
    def audit_fake(_path, receipt, **_kwargs):
        return {'return': receipt['return'], 'success': True, 'native_steps': 52, 'unique_positions': 52,
                'mismatching_pairs': 0, 'repeat_mismatching_pairs': 0, 'layout': layout.copy(),
                'counts': receipt['counts'], 'timings': {'native_seconds': .0001},
                'picker_counts': {'max_rowmass_abs': 0., 'first_phase': 26}}
    monkeypatch.setattr(report, 'audit_episode', audit_fake)
    report.write(evaluation/'started.json', {'automatic_retry': False, 'evaluation': recipe['evaluation'],
                 **{k+'_sha256': v for k, v in expected.items()}})
    report.write(evaluation/'all-fits-ready.json', {'status': 'complete', 'fits': maps})
    for name in report.NAMES:
        neural = name not in report.REFERENCES
        counts = {k: (1 if 'reset' in k or 'identity' in k else int(neural) if 'init' in k
                      else 52 if 'native' in k or 'decision' in k else 52*int(neural)) for k in work_names}
        for policy in report.POLICIES:
            folder = evaluation/'controllers'/name/policy; (folder/'episodes').mkdir(parents=True)
            value = {'A': .05, 'B': .1, 'C': .13}[policy]
            for index in range(64):
                stem = folder/'episodes'/f'{index:03d}'
                stem.with_suffix('.npz').write_bytes(b'fake array placeholder')
                report.write(stem.with_suffix('.json'), {'index': index, 'seed': index, 'return': value,
                    'protocol_sha256': expected['protocol'], 'inputs_sha256': expected['inputs'],
                    'checkpoint_sha256': 'b'*64 if neural else None, 'layout_sha256': digest,
                    'array_bytes': 1, 'whole_episode_seconds': .001, 'wall_seconds': .0001,
                    'construction_seconds': .0001, 'close_seconds': .0001, 'serialization_seconds': .0001,
                    'counts': counts, 'matched_pairs': 26})
                for k, v in counts.items(): totals[k] += v
            report.write(folder/'completed.json', {'status': 'complete', 'controller': name, 'policy': policy,
                         'episodes': 64, 'mean_return': value, 'successes': 64, 'native_steps': 64*52,
                         'whole_episode_seconds': .064, 'controller_shared_restore_seconds': .01})
    files = {str(p.relative_to(evaluation)): {'sha256': report.sha(p), 'bytes': p.stat().st_size}
             for p in evaluation.rglob('*') if p.is_file()}
    assert set(files) == report.expected_members()
    if corrupt_counter: totals['write_returned'] -= 1
    completed = {'status': 'complete', 'controllers': 20, 'episodes': 3840, 'policies': list(report.POLICIES),
                 'model_restores': 18, 'new_fits': 0, 'new_optimizer_steps': 0, 'native_replay_calls': 0,
                 'lineage': {}, **{k+'_sha256': v for k,v in expected.items()},
                 'started_sha256': report.sha(evaluation/'started.json'), 'restored_models': restored,
                 'counts': totals, 'uncompressed_array_bytes': 3840, 'layout_sha256_by_case': {str(i): digest for i in range(64)},
                 'distinct_layouts': 1, 'payload_bytes': sum(v['bytes'] for v in files.values()), 'wall_seconds': 6.,
                 'controller_wall_seconds': dict.fromkeys(report.NAMES, .25), 'final_hashing_seconds': .2, 'files': files}
    report.write(evaluation/'completed.json', completed)
    kwargs = {'root': root, **source_files, **{'expected_'+k+'_sha256': v for k,v in expected.items()},
              'evaluation': evaluation, 'expected_completed_sha256': report.sha(evaluation/'completed.json'), 'out': root/'report'}
    if corrupt_counter:
        with pytest.raises(ValueError, match='counters differ'):
            report.report(**kwargs)
        assert (root/'report/failed.json').exists() and not (root/'report/receipt.json').exists()
    else:
        result = report.report(**kwargs)
        assert result['status'] == 'complete' and result['continuation_passed']
        assert report.read(root/'report/summary.json')['coverage']['episodes'] == 3840
        with pytest.raises(FileExistsError):
            report.report(**kwargs)


def test_synthetic_figures_and_fixed_public_frame(report, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/visualize_card_controllers.py'
    spec = importlib.util.spec_from_file_location('visualize_card_controllers_test', path)
    visual = importlib.util.module_from_spec(spec); spec.loader.exec_module(visual)
    rows = fake_rows(report)
    families, contrasts, gate = report.aggregate(rows)
    visual.plot({'families': families, 'per_fit': rows, 'contrasts': contrasts, 'continuation_gate': gate}, tmp_path)
    assert all((tmp_path/('controllers.'+ext)).stat().st_size > 0 for ext in ('png','svg','pdf'))
    episode = {'actions': np.array([0], dtype=np.int64), 'rewards': np.array([0.]),
               'observations': np.full((2,52),13,dtype=np.int64)}
    frame = visual.replay_frame([episode]*3, [{'seed': 0}]*3, 1)
    assert frame.size == (1280,390)
    frame.save(tmp_path/'synthetic-public-frame.png')
    replay = visual.public_replay([episode]*3, [{'seed': 0}]*3, tmp_path/'synthetic-public.gif')
    assert replay['frames'] == 2 and replay['controller'] == 'gated_delta-pair0'
    assert replay['policies'] == ['A', 'B', 'C'] and replay['outcome_selected'] is False
