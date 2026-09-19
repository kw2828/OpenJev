"""Small constructed summaries/public frames only; no native/model/outcome reads."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

PATH = Path(__file__).resolve().parents[1] / 'scripts/visualize_card_calibration.py'
SPEC = importlib.util.spec_from_file_location('calibration_visual_test', PATH)
visual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visual)
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    import report_card_calibration as report


def synthetic():
    per_fit = {}
    for policy in visual.POLICIES:
        names = report.NAMES + report.REFERENCES if policy == 'baseline' else report.NAMES
        per_fit[policy] = {name: {'mean_return': 2 / 52, 'per_seed_returns': [2 / 52] * 64,
                                'per_seed_successes': [False] * 64, 'native_steps': 128, 'wall_seconds': 1.,
                                'beta': 2. if policy == 'temperature' else None} for name in names}
    scores = {name: {policy: {'all': {'eligible_episodes': 64, 'query_cards': 64,
                'nll': None if policy == 'hard' else (1. if policy == 'baseline' else .9),
                'nll_is_infinite': policy == 'hard', 'infinite_nll_queries': int(policy == 'hard'),
                'brier': .1, 'accuracy': .9}} for policy in visual.POLICIES} for name in report.NAMES}
    families, contrasts, transfer, gate = report.aggregate(per_fit, scores)
    data = {'status': 'complete', 'per_fit': per_fit, 'families': families, 'contrasts': contrasts,
            'transfer': transfer, 'continuation_gate': gate, 'fresh_baseline_prefix_scores': scores,
            'references': {name: per_fit['baseline'][name] for name in report.REFERENCES},
            'references_evaluated_once': True, 'coverage': {'native_games': 3584, 'native_rows': 56},
            'original_architecture_gate': {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True}}
    episodes, receipts = [], []
    for policy in visual.POLICIES:
        obs = np.full((3, 52), 13, np.int64)
        obs[1, 0] = 0; obs[2, :2] = 0
        episodes.append({'observations': obs, 'actions': np.array([0, 1], np.int64),
                         'rewards': np.array([0., 2 / 52], np.float64),
                         'terminated': np.zeros(2, bool), 'truncated': np.zeros(2, bool)})
        receipts.append({'status': 'complete', 'controller': 'kalman-pair0', 'index': 0, 'policy': policy,
                         'seed': 410, 'layout_sha256': 'a' * 64, 'tracker_policy': 'C', 'native_steps': 2,
                         'return': 2 / 52, 'calibration_completed_sha256': 'c' * 64,
                         'beta': 2. if policy == 'temperature' else None})
    return data, episodes, receipts


def test_actual_png_pdf_svg_and_all_public_gif_frames(tmp_path):
    data, episodes, receipts = synthetic()
    assert not data['continuation_gate']['passed'] and data['transfer']['hard']['nll_is_infinite']
    visual.plot(data, tmp_path)
    with Image.open(tmp_path / 'calibration.png') as png:
        assert png.width > 2000 and png.height > 800
    assert (tmp_path / 'calibration.pdf').read_bytes().startswith(b'%PDF')
    assert '<svg' in (tmp_path / 'calibration.svg').read_text()
    replay = visual.public_replay(episodes, receipts, tmp_path / 'public.gif')
    assert replay['controller'] == 'kalman-pair0' and replay['case_index'] == 0
    assert replay['frames'] == 3 and replay['public_only'] and not replay['outcome_selected']
    with Image.open(tmp_path / 'public.gif') as gif:
        assert gif.n_frames == 3 and gif.size == (1280, 390)
        for i in range(gif.n_frames):
            gif.seek(i)
            assert gif.info['duration'] == 160


def tree(tmp_path, monkeypatch):
    """Stub only prior full-tree hashing; exercise renderer report/arithmetic/preset checks."""
    data, episodes, receipts = synthetic()
    root, folder, evaluation = tmp_path, tmp_path / 'report', tmp_path / 'evaluation'
    folder.mkdir(); evaluation.mkdir()
    source = root / 'scripts/visualize_card_calibration.py'
    source.parent.mkdir(); source.write_bytes(PATH.read_bytes())
    monkeypatch.setattr(visual, '__file__', str(source))
    protocol = root / 'protocol.json'
    protocol.write_text(json.dumps({'bindings_file': 'bindings.json',
        'preset_visualization': {'controller': 'kalman-pair0', 'case_index': 0, 'selection': 'synthetic fixed fixture'}}))
    bindings = root / 'bindings.json'
    bindings.write_text(json.dumps({'files': {'scripts/visualize_card_calibration.py': visual.sha(source),
                                             'protocol.json': visual.sha(protocol)}}))
    data['bindings'] = {'source_sha256': 'd' * 64, 'protocol_sha256': visual.sha(protocol),
                        'evaluation_completed_sha256': 'e' * 64}
    (folder / 'summary.json').write_text(json.dumps(data))
    (folder / 'report.md').write_text('fabricated layout fixture, no empirical result')
    receipt = {'status': 'complete', 'source_sha256': 'd' * 64, 'new_model_calls': 0, 'new_native_calls': 0,
               'evaluation_completed_sha256': 'e' * 64, 'protocol_sha256': visual.sha(protocol),
               'bindings_sha256': visual.sha(bindings), 'calibration_completed_sha256': 'c' * 64,
               'continuation_passed': data['continuation_gate']['passed'],
               'files': {name: {'sha256': visual.sha(folder / name), 'bytes': (folder / name).stat().st_size}
                         for name in ('summary.json', 'report.md')}}
    for policy, arrays, item in zip(visual.POLICIES, episodes, receipts, strict=True):
        stem = evaluation / 'controllers/kalman-pair0' / policy / 'episodes/000'
        stem.parent.mkdir(parents=True)
        np.savez_compressed(stem.with_suffix('.npz'), **arrays)
        item['npz_sha256'] = visual.sha(stem.with_suffix('.npz'))
        stem.with_suffix('.json').write_text(json.dumps(item))
    monkeypatch.setattr(visual, 'reporter', lambda digest: SimpleNamespace(
        authenticate_tree=lambda *args: {'protocol_sha256': visual.sha(protocol)},
        expected_members=report.expected_members, aggregate=report.aggregate))
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    return root, protocol, folder, evaluation


def test_summary_arithmetic_and_failed_gate_remain_bound(tmp_path, monkeypatch):
    root, protocol, folder, evaluation = tree(tmp_path, monkeypatch)
    data, _, _, receipt = visual.authenticate(root, protocol, folder, visual.sha(folder / 'receipt.json'), evaluation)
    assert receipt['continuation_passed'] is False and data['continuation_gate']['total_checks'] == 12
    data['families']['baseline']['kalman']['mean_return'] += .1
    (folder / 'summary.json').write_text(json.dumps(data))
    receipt['files']['summary.json'] = {'sha256': visual.sha(folder / 'summary.json'), 'bytes': (folder / 'summary.json').stat().st_size}
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='numerical/gate'):
        visual.authenticate(root, protocol, folder, visual.sha(folder / 'receipt.json'), evaluation)


def test_fixed_replay_identity_rejects_chosen_other_case(tmp_path, monkeypatch):
    root, protocol, folder, evaluation = tree(tmp_path, monkeypatch)
    path = evaluation / 'controllers/kalman-pair0/temperature/episodes/000.json'
    item = json.loads(path.read_text()); item['index'] = 7; path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='Fixed first-case'):
        visual.authenticate(root, protocol, folder, visual.sha(folder / 'receipt.json'), evaluation)
