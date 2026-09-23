"""Fabricated presentation coverage and provenance; no empirical arrays or fitting."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    '_test_interruption_plot', ROOT / 'scripts/plot_otto_action_focused_interruption.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)
FIXTURE_PATH = ROOT / 'tests/test_plot_otto_action_focused.py'
assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == 'bf84be62c512a336f4b62006988aada4b22febd968fd1189dfd833c2dfa96bf4'
FIXTURE_SPEC = importlib.util.spec_from_file_location('_qualified_action_plot_fixtures', FIXTURE_PATH)
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


def aggregates():
    producer, old = fixtures.aggregates()
    gates = {name: {**gate, 'eligible': False, 'reason': 'Original parent unavailable'}
             for name, gate in producer['gates'].items()}
    summary = {**copy.deepcopy(producer), 'gates': copy.deepcopy(gates), 'technical_complete': False,
               'eligible_for_continuation': False, 'original_training_process_closure': False}
    result = {'version': plot.DIAGNOSTIC_VERSION, 'arithmetic_agreement': True,
        'original_training_process_closure': False, 'original_process_closure': False,
        'technical_complete': False, 'eligible_for_continuation': False, 'original_continuation_eligible': False,
        'original_audit': 'unavailable_not_replaced', 'scientific_calls': {'model': 0, 'native': 0, 'optimizer': 0, 'teacher': 0},
        'producer_summary': copy.deepcopy(producer), 'summary': summary, 'gates': gates,
        'fits': old['fits'], 'counts': {**old['counts'], 'required_passed': 0, 'checks': 1000},
        'worker_reported_wall_seconds': 204., 'limitations': ['Fabricated saved values only.']}
    return producer, result


def test_all_twelve_fits_and_scopes_remain_descriptive_without_changing_values():
    summary, result = aggregates()
    before = copy.deepcopy(result)
    assert plot.checked_summary(result) == summary
    data = plot.tables(summary, result)
    assert {k: len(v) for k, v in data.items()} == {'forecast_metrics': 9360, 'regime_metrics': 78,
        'family_means': 24, 'objective_contrasts': 36, 'objective_contrast_means': 12, 'costs': 12}
    for rows in data.values():
        assert all(r['eligible_for_continuation'] is False and r['evidence_scope'] == plot.EVIDENCE_SCOPE for r in rows)
    assert {r['seed'] for r in data['costs']} == set(plot.original.SEEDS)
    assert all(r['timing_scope'] == plot.TIMING_SCOPE for r in data['costs'])
    assert result == before


@pytest.mark.parametrize('field', ['original_training_process_closure', 'original_process_closure',
    'technical_complete', 'eligible_for_continuation', 'original_continuation_eligible'])
def test_any_original_qualification_claim_is_rejected(field):
    _, result = aggregates()
    result[field] = True
    with pytest.raises(ValueError, match='never qualifies'):
        plot.checked_summary(result)


@pytest.mark.parametrize('defect', ['technical_row', 'eligible_gate', 'missing_gate', 'changed_summary', 'calls'])
def test_summary_or_gate_misrepresentation_is_rejected(defect):
    _, result = aggregates()
    if defect == 'technical_row':
        result['producer_summary']['required'][0]['passes'] = True
    elif defect == 'eligible_gate':
        result['gates']['architecture']['eligible'] = True
    elif defect == 'missing_gate':
        result['gates'].pop('architecture')
    elif defect == 'changed_summary':
        result['summary']['required_passed'] += 1
    else:
        result['scientific_calls']['model'] = 1
    with pytest.raises(ValueError):
        plot.checked_summary(result)


def test_all_three_figures_render_with_banner_and_worker_reported_cost_labels(tmp_path, monkeypatch):
    summary, result = aggregates()
    runner = plot.Plot(SimpleNamespace(output=tmp_path))
    captured = {}

    def saved(_self, fig, stem):
        text = [t.get_text() for t in fig.texts]
        text += [ax.yaxis.label.get_text() for ax in fig.axes]
        buffer = io.BytesIO()
        fig.savefig(buffer, format='svg', bbox_inches='tight')
        captured[stem] = (text, buffer.getvalue())

    monkeypatch.setattr(plot.original.Plot, 'save_figure', saved)
    runner.figures(summary, plot.tables(summary, result))
    assert set(captured) == set(plot.original.FIGURES)
    for text, svg in captured.values():
        assert text.count(plot.BANNER) == 1 and plot.BANNER.encode() in svg
    cost_text = captured[plot.original.FIGURES[2]][0]
    assert 'Worker-reported fitting time, not deployment latency' in cost_text
    assert 'Within-seed worker-reported time ratio' in cost_text
    assert not any('Measured' in text or 'measured fit' in text or 'authenticated saved measurements' in text for text in cost_text)
    assert any('parent bounds, exit and reaping remain unverified' in text for text in cost_text)


def closed_fixture(tmp_path, monkeypatch):
    for module in (plot, plot.original, plot.original.base):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    summary, result = aggregates()
    directory, output = tmp_path / 'diagnostic', tmp_path / 'plot'
    directory.mkdir(); output.mkdir()
    source = tmp_path / plot.DIAGNOSTIC
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fabricated diagnostic source')
    source_pin = fixtures.descriptor(source)['sha256']
    monkeypatch.setattr(plot, 'DIAGNOSTIC_PIN', source_pin)
    inputs = {}
    for role in ('training_plan', 'worker', 'launch', 'interruption', 'engineering', 'launcher_engineering'):
        inputs[role] = fixtures.save(tmp_path / (role + '.json'), {'fabricated_role': role})
    limits = {'seconds': 240, 'rss_bytes': 2 * 1024**3, 'output_bytes': 256 * 1024**2}
    plan_path = tmp_path / 'plan.json'
    plan = {'version': plot.DIAGNOSTIC_VERSION, 'status': 'frozen_before_saved_array_decode',
        'original_process_closure': False, 'technical_complete': False, 'original_audit': 'unavailable_not_replaced',
        'sources': {plot.DIAGNOSTIC: source_pin}, 'inputs': inputs, 'limits': limits,
        'original_terminal_path': str(tmp_path / 'missing-original-training.terminal.json')}
    plan_pin = fixtures.save(plan_path, plan)
    worker, terminal, launch = fixtures.phase(tmp_path, directory, plot.DIAGNOSTIC, 240,
        {'--plan': str(plan_path), '--plan-sha256': plan_pin['sha256'], '--output': str(directory)}, mode='run')
    fixtures.save(directory / 'started.json', {'inputs': inputs, 'started_ns': worker['started_ns'],
        'original_process_closure': False, 'technical_complete': False, 'launch': launch})
    fixtures.save(directory / 'diagnostic.json', result)
    worker.update(version=plot.DIAGNOSTIC_VERSION, status='completed', arithmetic_agreement=True, failures=[],
        requires_successful_original_diagnostic_supervisor=True, original_process_closure=False, technical_complete=False,
        original_audit='unavailable_not_replaced', plan_sha256=plan_pin['sha256'], sources=plan['sources'], inputs=inputs,
        limits=limits, model_calls=0, native_calls=0, optimizer_calls=0, teacher_calls=0, counts=result['counts'],
        files={name: fixtures.descriptor(directory / name) for name in ('started.json', 'diagnostic.json')})
    receipt_pin = fixtures.save(directory / 'receipt.json', worker)
    terminal_path = tmp_path / 'diagnostic.terminal.json'
    terminal_pin = fixtures.save(terminal_path, terminal)
    args = SimpleNamespace(output=output, diagnostic_plan=plan_path, diagnostic_plan_sha256=plan_pin['sha256'],
        diagnostic_directory=directory, diagnostic_receipt_sha256=receipt_pin['sha256'],
        diagnostic_terminal=terminal_path, diagnostic_terminal_sha256=terminal_pin['sha256'])
    return plot.Plot(args), summary, result


def test_actual_new_diagnostic_parent_is_closed_without_loading_original_trial_files(tmp_path, monkeypatch):
    runner, expected_summary, expected_result = closed_fixture(tmp_path, monkeypatch)
    # The original plot authenticate would require an original trial terminal and audit.json.
    monkeypatch.setattr(plot.original.Plot, 'authenticate', lambda *_: pytest.fail('Original trial audit is unavailable'))
    summary, result = runner.authenticate()
    assert (summary, result) == (expected_summary, expected_result)
    assert runner.receipt['diagnostic_seconds'] == {'worker': 1., 'closed_parent': 3.}
    assert runner.receipt['worker_reported_training_seconds'] == 204.
    assert runner.receipt['original_training_process_closure'] is False
    assert not any(path.endswith('.npz') for path in runner.bound)


def test_post_body_check_counter_increase_accepts_same_saved_counts(tmp_path, monkeypatch):
    runner, expected_summary, expected_result = closed_fixture(tmp_path, monkeypatch)
    path = runner.args.diagnostic_directory / 'receipt.json'
    receipt = json.loads(path.read_text())
    receipt['counts']['checks'] += 316
    runner.args.diagnostic_receipt_sha256 = fixtures.save(path, receipt)['sha256']
    summary, result = runner.authenticate()
    assert (summary, result) == (expected_summary, expected_result)
    assert result['counts']['checks'] == 1000
    assert receipt['counts']['checks'] == 1316
    assert runner.receipt['eligible_for_continuation'] is False


@pytest.mark.parametrize('defect', ['decrease', 'replace_count', 'missing_count', 'extra_count', 'float_count'])
def test_counter_decrease_or_any_changed_noncheck_count_is_rejected(tmp_path, monkeypatch, defect):
    runner, _, _ = closed_fixture(tmp_path, monkeypatch)
    path = runner.args.diagnostic_directory / 'receipt.json'
    receipt = json.loads(path.read_text())
    receipt['counts']['checks'] += 316
    if defect == 'decrease':
        receipt['counts']['checks'] = 999
    elif defect == 'replace_count':
        receipt['counts']['collection_episodes'] += 1
    elif defect == 'missing_count':
        receipt['counts'].pop('collection_episodes')
    elif defect == 'extra_count':
        receipt['counts']['invented_count'] = 0
    else:
        receipt['counts']['collection_episodes'] = float(receipt['counts']['collection_episodes'])
    runner.args.diagnostic_receipt_sha256 = fixtures.save(path, receipt)['sha256']
    with pytest.raises(ValueError, match='counter cannot decrease|exact non-check'):
        runner.authenticate()


@pytest.mark.parametrize('side', ['body', 'receipt'])
@pytest.mark.parametrize('value', [-1, 1000.0, True, '1000', None])
def test_check_counter_requires_nonnegative_integer_on_both_sides(side, value):
    counts = {'body': {'checks': 1000, 'fits': 12}, 'receipt': {'checks': 1000, 'fits': 12}}
    counts[side]['checks'] = value
    before = copy.deepcopy(counts)
    with pytest.raises(ValueError, match='nonnegative integer'):
        plot.checked_counts(counts['body'], counts['receipt'])
    assert counts == before


@pytest.mark.parametrize('side', ['body', 'receipt'])
def test_missing_check_counter_is_rejected(side):
    counts = {'body': {'checks': 0, 'fits': 12}, 'receipt': {'checks': 0, 'fits': 12}}
    counts[side].pop('checks')
    with pytest.raises(ValueError, match='require diagnostic check counters'):
        plot.checked_counts(counts['body'], counts['receipt'])


def test_zero_check_counter_is_valid_and_inputs_are_unchanged():
    body, receipt = {'checks': 0, 'fits': 12}, {'checks': 0, 'fits': 12}
    before = copy.deepcopy((body, receipt))
    plot.checked_counts(body, receipt)
    assert (body, receipt) == before


@pytest.mark.parametrize('defect', ['timeout', 'unreaped', 'wrong_script', 'source', 'payload', 'extra_file', 'original_terminal'])
def test_bad_diagnostic_closure_fails_before_saved_outcome_decode(tmp_path, monkeypatch, defect):
    runner, _, _ = closed_fixture(tmp_path, monkeypatch)
    if defect in ('timeout', 'unreaped', 'wrong_script'):
        value = json.loads(runner.args.diagnostic_terminal.read_text())
        if defect == 'timeout':
            value['timed_out'] = True
        elif defect == 'unreaped':
            value['cleanup']['reaped'] = False
        else:
            value['command'][2] = str(tmp_path / 'wrong.py')
        runner.args.diagnostic_terminal_sha256 = fixtures.save(runner.args.diagnostic_terminal, value)['sha256']
    elif defect == 'source':
        (tmp_path / plot.DIAGNOSTIC).write_bytes(b'changed')
    elif defect == 'payload':
        (runner.args.diagnostic_directory / 'diagnostic.json').write_bytes(b'changed')
    elif defect == 'extra_file':
        (runner.args.diagnostic_directory / 'extra.json').write_text('{}')
    else:
        (tmp_path / 'missing-original-training.terminal.json').write_text('{}')
    observed, original_read = [], runner.read

    def read(path):
        observed.append(path.name)
        return original_read(path)

    monkeypatch.setattr(runner, 'read', read)
    with pytest.raises(ValueError):
        runner.authenticate()
    assert 'diagnostic.json' not in observed


def test_failed_publication_retains_exclusive_failure_receipt_and_no_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, 'ROOT', tmp_path)
    runner = plot.Plot(SimpleNamespace(output=tmp_path / 'failed'))
    monkeypatch.setattr(runner, 'bind', lambda *_: None)
    monkeypatch.setattr(runner, 'descriptor', lambda path: {'path': str(path), 'sha256': 'a' * 64,
        'bytes': path.stat().st_size if path.exists() else 0})

    def failed_authentication():
        raise ValueError('fabricated invalid original diagnostic parent')

    monkeypatch.setattr(runner, 'authenticate', failed_authentication)
    with pytest.raises(ValueError, match='fabricated invalid'):
        runner.execute()
    receipt = json.loads((runner.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['eligible_for_continuation'] is False
    assert receipt['technical_complete'] is receipt['original_training_process_closure'] is False
    assert receipt['model_calls'] == receipt['optimizer_calls'] == receipt['array_decodes'] == 0
    with pytest.raises(FileExistsError):
        runner.execute()
