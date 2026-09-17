"""Publication checks use synthetic aggregates, without model or engine calls."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('matplotlib')

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_publish_chess_student',
                                             ROOT/'scripts/publish_chess_student.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def summary_fixture():
    fits = []
    for mode, counts in [('gru', [148, 148, 150]), ('circuit', [145, 148, 148]), ('rewired', [149, 144, 143])]:
        for seed, correct in zip(publisher.SEEDS, counts, strict=True):
            fits.append({'mode': mode, 'seed': seed, 'examples': 1024,
                         'top1_teacher_agreement': correct/1024, 'target_nll': 3.,
                         'value_mae': .57, 'value_mse': .45})
    means = {mode: {metric: float(np.mean([row[metric] for row in fits if row['mode'] == mode]))
                    for metric in publisher.METRICS} for mode in publisher.MODES}
    return {'status': 'completed', 'protocol': copy.deepcopy(publisher.STUDY.PROTOCOL),
            'fits': fits, 'mean_by_mode': means, 'novelty_established': False, 'elo_estimate': None,
            'data_role': 'generated development set, not untouched confirmation',
            'baselines': {'uniform_random': {'expected_top1_teacher_agreement': .045},
                          'training_move_frequency': {'top1_teacher_agreement': .071}},
            'continuation_gate': publisher.STUDY.continuation_gate(fits), 'plan_sha256': 'a'*64}


def test_all_seed_points_and_failed_gate_are_visible():
    value = summary_fixture()
    publisher.validate_summary(value)
    fig = publisher.figure(value)
    assert len(fig.axes[0].collections) == 9
    assert len(fig.axes[1].collections) == 4
    assert fig.axes[0].get_ylim()[0] == 0
    assert any('FAILED' in text.get_text() for text in fig.texts)
    publisher.plt.close(fig)


@pytest.mark.parametrize('change', ['missing_fit', 'duplicate_fit', 'mean', 'gate', 'scope', 'nonfinite'])
def test_inconsistent_or_overstated_aggregates_fail(change):
    value = summary_fixture()
    if change == 'missing_fit':
        value['fits'].pop()
    elif change == 'duplicate_fit':
        value['fits'][0] = copy.deepcopy(value['fits'][1])
    elif change == 'mean':
        value['mean_by_mode']['circuit']['top1_teacher_agreement'] = .20
    elif change == 'gate':
        value['continuation_gate']['passed'] = True
    elif change == 'scope':
        value['novelty_established'] = True
    else:
        value['fits'][0]['target_nll'] = float('nan')
    with pytest.raises(ValueError):
        publisher.validate_summary(value)


def test_report_requires_matching_completion_receipt(tmp_path):
    summary = summary_fixture()
    path = tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    receipt = {'status': 'completed', 'summary_sha256': publisher.sha(path), 'plan_sha256': 'a'*64}
    (tmp_path/'completed.json').write_text(json.dumps(receipt))
    assert publisher.load_report(tmp_path)[0] == summary
    path.write_text(json.dumps({**summary, 'extra': 'changed after completion'}))
    with pytest.raises(ValueError, match='checksum'):
        publisher.load_report(tmp_path)


def test_figure_receipt_hashes_clean_svg_and_prevents_overwrite(tmp_path):
    summary = summary_fixture()
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'completed.json').write_text('{}')
    paths = publisher.save_figure(summary, tmp_path, tmp_path/'figure')
    receipt = json.loads(paths['json'].read_text())
    assert receipt['plots']['png'] == publisher.sha(paths['png'])
    assert receipt['plots']['svg'] == publisher.sha(paths['svg'])
    assert all(line.rstrip() == line for line in paths['svg'].read_text().splitlines())
    with pytest.raises(FileExistsError):
        publisher.save_figure(summary, tmp_path, tmp_path/'figure')
