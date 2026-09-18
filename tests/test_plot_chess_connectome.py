"""Synthetic invented values only; no chess observations or measured outcomes."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('plot_connectome', ROOT/'scripts/plot_chess_connectome.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture_summary():
    summary = {'status': 'completed', 'plan_sha256': 'a'*64, 'execution_receipt_sha256': 'b'*64,
               'metrics': {}, 'engine_loss': {}, 'latency': {}, 'training_costs': {},
               'grading_costs': {}, 'backbone_pretraining_costs': {}, 'data_costs': {},
               'wall_seconds': 123., 'limits': [f'Synthetic limitation {i}' for i in range(7)],
               'report_new_model_or_engine_calls': 0, 'synthetic_fixture': True}
    for vi, variant in enumerate(plot.VARIANTS):
        for si, seed in enumerate(plot.SEEDS):
            identity = f'{variant}-{seed}'
            summary['metrics'][identity], summary['engine_loss'][identity] = {}, {}
            for split in plot.SPLITS:
                correct = 240+vi*12+si*5+(10 if split == 'dev' else 0)
                summary['metrics'][identity][split] = {
                    'examples': 2048, 'correct': correct, 'agreement': correct/2048,
                    'target_nll': 2.5, 'value_mae': .3, 'mean_confidence': .2,
                    'mismatch_confidence': .18, 'entropy': 3., 'hidden_rms': .1, 'logit_span': 3.}
                summary['engine_loss'][identity][split] = (.07 if variant == 'biological' else .2)+si*.005
            ms = 1+vi*.3+si*.05
            summary['latency'][identity] = {'mean_ms': ms, 'total_wall_ms': ms*128, 'warmup_wall_ms': 3*ms,
                                           'host_load_average_before_latency': [1., 2., 3.],
                                           'host_load_average_after_latency': [2., 3., 4.]}
            if variant != 'direct':
                summary['training_costs'][identity] = {}
    summary['topology_comparison'] = plot.expected_comparison(summary['engine_loss'])
    return summary


def write_report(tmp_path, summary=None):
    directory = tmp_path/'report'
    directory.mkdir()
    path = directory/'summary.json'
    summary = fixture_summary() if summary is None else summary
    path.write_text(json.dumps(summary))
    digest = plot.sha256(path.read_bytes())
    (directory/'completed.json').write_text(json.dumps({'status': 'completed',
        'plan_sha256': summary['plan_sha256'], 'files': {'summary.json': digest}}))
    return path, digest


def test_all_models_seeds_panels_and_descriptive_means():
    summary = fixture_summary()
    assert plot.validate(summary, synthetic_fixture=True) is summary
    values = plot.chart_values(summary)
    assert [r['variant'] for r in values] == list(plot.VARIANTS)
    assert {s['model'] for r in values for s in r['seeds']} == set(plot.IDENTITIES)
    for record in values:
        assert [s['seed'] for s in record['seeds']] == list(plot.SEEDS)
        for split in plot.SPLITS:
            expected = math.fsum(s['target_agreement_percent'][split] for s in record['seeds'])/3
            assert record['mean']['target_agreement_percent'][split] == expected


def test_gate_matches_independent_manual_expectation():
    summary = fixture_summary()
    gate = summary['topology_comparison']
    assert gate['continuation_passed'] is True
    assert len(gate['checks']) == 30
    assert gate['means']['biological']['dev'] == pytest.approx(.075)
    summary['engine_loss']['rewire163-109']['shift'] = .075
    summary['topology_comparison'] = plot.expected_comparison(summary['engine_loss'])
    assert summary['topology_comparison']['continuation_passed'] is False
    plot.validate(summary, synthetic_fixture=True)


@pytest.mark.parametrize('corruption', [
    'missing_model', 'extra_model', 'missing_seed_loss', 'missing_panel', 'extra_panel',
    'wrong_count', 'bool_count', 'wrong_agreement', 'nan_agreement', 'infinite_loss',
    'bool_loss', 'out_of_bound_loss', 'negative_latency', 'wrong_latency', 'bad_host_load',
    'wrong_gate', 'bool_gate', 'wrong_mean', 'missing_gate_check', 'missing_fit',
    'incomplete', 'new_inference', 'bad_hash', 'missing_limit', 'extra_field', 'wrong_confidence',
])
def test_bad_saved_values_fail_closed(corruption):
    s = fixture_summary()
    if corruption == 'missing_model':
        s['metrics'].pop('direct-97')
    elif corruption == 'extra_model':
        s['metrics']['selected-best'] = s['metrics']['direct-97']
    elif corruption == 'missing_seed_loss':
        s['engine_loss'].pop('biological-109')
    elif corruption == 'missing_panel':
        s['metrics']['direct-97'].pop('shift')
    elif corruption == 'extra_panel':
        s['engine_loss']['direct-97']['extra'] = 0.
    elif corruption == 'wrong_count':
        s['metrics']['direct-97']['dev']['examples'] = 2047
    elif corruption == 'bool_count':
        s['metrics']['direct-97']['dev']['correct'] = True
    elif corruption == 'wrong_agreement':
        s['metrics']['direct-97']['dev']['agreement'] += .001
    elif corruption == 'nan_agreement':
        s['metrics']['direct-97']['dev']['agreement'] = float('nan')
    elif corruption == 'infinite_loss':
        s['engine_loss']['direct-97']['dev'] = float('inf')
    elif corruption == 'bool_loss':
        s['engine_loss']['direct-97']['dev'] = True
    elif corruption == 'out_of_bound_loss':
        s['engine_loss']['direct-97']['dev'] = -2.1
    elif corruption == 'negative_latency':
        s['latency']['direct-97']['warmup_wall_ms'] = -.1
    elif corruption == 'wrong_latency':
        s['latency']['direct-97']['mean_ms'] += .1
    elif corruption == 'bad_host_load':
        s['latency']['direct-97']['host_load_average_before_latency'][1] = True
    elif corruption == 'wrong_gate':
        s['topology_comparison']['continuation_passed'] = False
    elif corruption == 'bool_gate':
        s['topology_comparison']['continuation_passed'] = 1
    elif corruption == 'wrong_mean':
        s['topology_comparison']['means']['biological']['shift'] += .1
    elif corruption == 'missing_gate_check':
        s['topology_comparison']['checks'].pop()
    elif corruption == 'missing_fit':
        s['training_costs'].pop('biological-127')
    elif corruption == 'incomplete':
        s['status'] = 'started'
    elif corruption == 'new_inference':
        s['report_new_model_or_engine_calls'] = 1
    elif corruption == 'bad_hash':
        s['execution_receipt_sha256'] = 'not-a-hash'
    elif corruption == 'missing_limit':
        s['limits'].pop()
    elif corruption == 'extra_field':
        s['selected_winner'] = 'biological-97'
    elif corruption == 'wrong_confidence':
        s['metrics']['direct-97']['dev']['mean_confidence'] = 1.01
    with pytest.raises(ValueError):
        plot.validate(s, synthetic_fixture=True)


def test_fixture_requires_label_and_explicit_option(tmp_path):
    path, digest = write_report(tmp_path)
    with pytest.raises(ValueError, match='fixture label'):
        plot.load_completed(path, digest)
    summary = fixture_summary()
    summary.pop('synthetic_fixture')
    with pytest.raises(ValueError):
        plot.validate(summary, synthetic_fixture=True)
    assert plot.validate(summary) == summary


@pytest.mark.parametrize('corruption', ['source_hash', 'receipt_hash', 'receipt_plan', 'extra_file', 'missing_receipt'])
def test_completed_report_binding(tmp_path, corruption):
    path, digest = write_report(tmp_path)
    receipt_path = path.parent/'completed.json'
    receipt = json.loads(receipt_path.read_text())
    if corruption == 'source_hash':
        path.write_text(path.read_text()+'\n')
    elif corruption == 'receipt_hash':
        receipt['files']['summary.json'] = 'c'*64
        receipt_path.write_text(json.dumps(receipt))
    elif corruption == 'receipt_plan':
        receipt['plan_sha256'] = 'c'*64
        receipt_path.write_text(json.dumps(receipt))
    elif corruption == 'extra_file':
        (path.parent/'extra.json').write_text('{}')
    elif corruption == 'missing_receipt':
        receipt_path.unlink()
    with pytest.raises(ValueError):
        plot.load_completed(path, digest, synthetic_fixture=True)


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        plot._json('{"status":"started","status":"completed"}')


def test_signed_negative_losses_are_retained_and_gate_can_fail():
    summary = fixture_summary()
    summary['engine_loss']['biological-97']['dev'] = -.1
    summary['engine_loss']['rewire151-97']['dev'] = -.7
    summary['topology_comparison'] = plot.expected_comparison(summary['engine_loss'])
    plot.validate(summary, synthetic_fixture=True)
    assert not summary['topology_comparison']['continuation_passed']
    fig = plot.figure(summary, synthetic_fixture=True)
    try:
        assert fig.axes[2].get_ylim()[0] < -.7
        assert fig.axes[2].get_ylim() == fig.axes[3].get_ylim()
        assert len(fig.axes) == 5
        assert any('DID NOT PASS' in t.get_text() for t in fig.texts)
        assert 'SYNTHETIC FIXTURE - NOT RESULTS' in fig._suptitle.get_text()
        for ax in fig.axes:
            assert [len(c.get_offsets()) for c in ax.collections] == [7, 7, 7, 7]
    finally:
        plot.plt.close(fig)


def test_render_exclusive_png_and_traceable_manifest(tmp_path):
    path, digest = write_report(tmp_path)
    out = tmp_path/'synthetic-plot-only'
    manifest = plot.render(path, digest, out, synthetic_fixture=True)
    assert {p.name for p in out.iterdir()} == {'connectome-chess.png', 'source-manifest.json'}
    assert manifest == json.loads((out/'source-manifest.json').read_text())
    assert manifest['synthetic_fixture'] is True
    assert manifest['summary_sha256'] == digest
    assert manifest['figure_sha256']['connectome-chess.png'] == plot.sha256((out/'connectome-chess.png').read_bytes())
    assert (out/'connectome-chess.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert manifest['new_model_or_engine_calls'] == 0
    assert manifest['models'] == 21 and manifest['gate_checks_total'] == 30
    assert manifest['plotted_values'] == plot.chart_values(fixture_summary())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        plot.render(path, digest, out, synthetic_fixture=True)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_input_is_unchanged_by_plotting():
    summary = fixture_summary()
    before = copy.deepcopy(summary)
    fig = plot.figure(summary, synthetic_fixture=True)
    plot.plt.close(fig)
    assert summary == before


def test_render_cannot_add_files_inside_authenticated_report(tmp_path):
    path, digest = write_report(tmp_path)
    with pytest.raises(ValueError, match='outside the authenticated'):
        plot.render(path, digest, path.parent/'figures', synthetic_fixture=True)
    assert {p.name for p in path.parent.iterdir()} == {'summary.json', 'completed.json'}
