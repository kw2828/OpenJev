"""Publisher integrity checks use synthetic aggregates, never benchmark examples."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('matplotlib')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_test_publish_codex_astra_text',
                                            ROOT / 'scripts' / 'publish_codex_astra_text.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def fixture_summary():
    protocol = copy.deepcopy(publisher.EXPECTED_PROTOCOL)
    summary = {'status': 'completed', 'protocol': protocol, 'plan_sha256': 'a' * 64,
               'source_packet_sha256': 'b' * 64, 'teacher_completed_sha256': 'c' * 64,
               'independent_confirmation': False, 'novelty_established': False,
               'direct_paid_api_calls': 0, 'codex_usage_cost': 'Unavailable',
               'teacher_training_label_agreement': {'boolq': 60 / 64, 'clinc_domains': 1.},
               'arms': {}, 'contrasts': {}, 'checks': {},
               'results_sha256': {f'{arm}/result-{seed}.json': 'd' * 64
                                  for arm in publisher.ARMS for seed in publisher.SEEDS}}
    for arm_index, arm in enumerate(publisher.ARMS):
        summary['arms'][arm] = {'reused_control': arm != 'astra_codex', 'training_seconds': 1., 'tasks': {}}
        for task in publisher.TASKS:
            rows = []
            denominator = 42 if task == 'boolq' else 8
            classes = ['no', 'yes'] if task == 'boolq' else [f'domain-{index}' for index in range(11)]
            for seed_index, seed in enumerate(publisher.SEEDS):
                correct = [20, 23, 30][arm_index] + seed_index if task == 'boolq' else [1, 4, 6][arm_index]
                accuracy = correct / denominator
                rows.append({'seed': seed, 'examples': publisher.COUNTS[task],
                             'accuracy': accuracy, 'macro_class_accuracy': accuracy,
                             'class_accuracy': dict.fromkeys(classes, accuracy),
                             'nll': .8, 'multiclass_brier': .6, 'median_latency_ms': 1., 'p95_latency_ms': 2.})
            summary['arms'][arm]['tasks'][task] = {
                'per_fit': rows, 'mean_across_fits': {key: float(np.mean([row[key] for row in rows]))
                                                    for key in publisher.METRICS}}
    for task in publisher.TASKS:
        summary['contrasts'][task] = {}
        for arm in ('untrained', 'gold'):
            delta = (summary['arms']['astra_codex']['tasks'][task]['mean_across_fits']['accuracy']
                     - summary['arms'][arm]['tasks'][task]['mean_across_fits']['accuracy'])
            summary['contrasts'][task][f'astra_codex_minus_{arm}'] = {
                'delta': delta, 'exploratory_95_percent_interval': [delta - .05, delta + .05]}
        summary['checks'][f'{task}_gain_vs_untrained'] = summary['contrasts'][task]['astra_codex_minus_untrained']['delta'] >= .1
        summary['checks'][f'{task}_within_gold_margin'] = summary['contrasts'][task]['astra_codex_minus_gold']['delta'] >= -.05
    summary['continuation_passed'] = all(summary['checks'].values())
    provenance = {key: copy.deepcopy(summary[key]) for key in ('protocol', 'plan_sha256', 'source_packet_sha256', 'results_sha256')}
    provenance.update(worker_packet_contents_published=False,
                      sources_sha256={name: publisher.sha(ROOT / name) for name in publisher.PRODUCER_SOURCES})
    return summary, provenance


def test_valid_aggregate_only_report_and_complete_three_fit_figure():
    summary, provenance = fixture_summary()
    publisher.validate_aggregates(summary, provenance)
    fig = publisher.figure(summary, fixture=True)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.collections) == 9  # Every fit is shown, including equal values.
        assert ax.get_ylim()[0] <= 0 and ax.get_ylim()[1] >= 100
        assert ax.get_title() == '' and ax.get_title(loc='left')
    assert any('SYNTHETIC TEST FIXTURE' in text.get_text() for text in fig.texts)
    assert any('not teacher evaluation accuracy' in text.get_text() for text in fig.texts)
    assert any('Reused public development' in text.get_text() for text in fig.texts)
    publisher.plt.close(fig)


@pytest.mark.parametrize('fault', ['missing_seed', 'duplicate_seed', 'mean', 'count', 'class_count', 'contrast',
                                  'gate', 'overall_gate', 'teacher_count', 'nonfinite', 'confirmation', 'control'])
def test_publisher_rejects_incomplete_inconsistent_or_overclaimed_aggregates(fault):
    summary, provenance = fixture_summary()
    target = summary['arms']['astra_codex']['tasks']['boolq']
    if fault == 'missing_seed':
        target['per_fit'].pop()
    elif fault == 'duplicate_seed':
        target['per_fit'][1]['seed'] = 17
    elif fault == 'mean':
        target['mean_across_fits']['accuracy'] += .01
    elif fault == 'count':
        target['per_fit'][0]['examples'] = 85
    elif fault == 'class_count':
        target['per_fit'][0]['class_accuracy']['no'] += .001
    elif fault == 'contrast':
        summary['contrasts']['boolq']['astra_codex_minus_gold']['delta'] += .01
    elif fault == 'gate':
        summary['checks']['boolq_gain_vs_untrained'] = False
    elif fault == 'overall_gate':
        summary['continuation_passed'] = False
    elif fault == 'teacher_count':
        summary['teacher_training_label_agreement']['boolq'] = .91234
    elif fault == 'nonfinite':
        target['per_fit'][0]['accuracy'] = float('nan')
    elif fault == 'confirmation':
        summary['independent_confirmation'] = True
    else:
        summary['arms']['gold']['reused_control'] = False
    with pytest.raises(ValueError):
        publisher.validate_aggregates(summary, provenance)


def test_loader_checks_only_allowlisted_sources_and_matching_provenance(tmp_path):
    summary, provenance = fixture_summary()
    for name, value in [('summary', summary), ('provenance', provenance)]:
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    assert publisher.load_report(tmp_path) == (summary, provenance)
    provenance['sources_sha256']['../do-not-open'] = 'e' * 64
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='producer sources'):
        publisher.load_report(tmp_path)
    provenance['sources_sha256'].pop('../do-not-open')
    provenance['sources_sha256']['scripts/codex_astra_distillation.py'] = 'f' * 64
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='source changed'):
        publisher.load_report(tmp_path)


def test_publisher_preserves_existing_metadata_but_never_replaces_existing_figures(tmp_path):
    path = tmp_path / 'plan-metadata.json'
    path.write_text('{"synthetic_fixture":true}\n')
    before = path.read_bytes()
    assert publisher.preserved_metadata(tmp_path) == {path.name: publisher.sha(path)}
    assert path.read_bytes() == before
    (tmp_path / 'student-results.png').write_bytes(b'synthetic figure placeholder')
    with pytest.raises(FileExistsError, match='preserved plan metadata'):
        publisher.preserved_metadata(tmp_path)
