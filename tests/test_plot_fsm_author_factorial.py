"""Small fabricated scalar and opaque-admission checks, never measured data."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('factorial_plot', ROOT/'scripts/plot_fsm_author_factorial.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def scalar_fixture():
    cells = {}
    for cell, scale, latency in zip(plot.CELLS, (1., .5, .25, .125), (40., 80., 30., 60.), strict=True):
        cells[cell] = {'rows': [{'record_id': name, 'status': 'complete', 'rmse': i*scale}
                               for i, name in enumerate(plot.admission.DEV)],
            'mean_rmse': 5.5*scale, 'amplitude_mean_rmse': {'100mV': 2.5*scale, '200mV': 8.5*scale},
            'median_request_ms': latency, 'persistent_numeric_bytes': 60184,
            'context_status_counts': {'ITERATION_CAP': 384},
            'failure_counts': dict.fromkeys(('forecast', 'record', 'warmup', 'timing'), 0)}
    continuation = {'candidate': 'unchanged_candidate', 'family_means': {'unchanged_candidate': .75},
                    'reference_complete': True, 'passed': 0, 'total': 4}
    result = {'matrix_status': 'FACTORIAL_COMPLETE', 'cells': cells, 'new_fit_status': 'FIT_ONLY_COMPLETE',
              'new_fit_iterations': 12345, 'continuation': continuation, 'contrasts': {'fixture': True}}
    summary = {'status': 'FACTORIAL_COMPLETE', 'new_fit_status': 'FIT_ONLY_COMPLETE', 'new_fit_iterations': 12345,
               'counts': dict(plot.COUNTS), 'continuation': copy.deepcopy(continuation), 'contrasts': {'fixture': True},
               'cells': {c: {k: copy.deepcopy(v) for k, v in row.items() if k != 'rows'} for c, row in cells.items()}}
    audit = {'status': 'PASS', 'agreement': True, 'scientific_status': 'REFERENCE_COMPLETE',
             'results': result, 'counts': {'request_replays': 1536}}
    return audit, summary


def test_hand_scalars_actual_iterations_and_no_candidate_latency():
    values = plot.extract(*scalar_fixture())
    assert [r['mean_rmse'] for r in values['cells']] == [5.5, 2.75, 1.375, .6875]
    assert [r['median_request_ms'] for r in values['cells']] == [40., 80., 30., 60.]
    assert all('diagnostic' in r['label'] for r in values['cells'][:2])
    assert all('12,345' in r['label'] for r in values['cells'][2:])
    assert values['candidate_rmse'] == .75 and 'candidate_latency' not in values


@pytest.mark.parametrize('change', ['audit', 'matrix', 'missing_cell', 'counts', 'metric', 'record', 'timing', 'failure'])
def test_reject_incomplete_or_inconsistent_scalars(change):
    audit, summary = scalar_fixture()
    if change == 'audit':
        audit['agreement'] = False
    elif change == 'matrix':
        audit['results']['matrix_status'] = summary['status'] = 'FACTORIAL_INCOMPLETE'
    elif change == 'missing_cell':
        del audit['results']['cells']['new64']
    elif change == 'counts':
        summary['counts']['timed_requests'] = 95
    elif change == 'metric':
        summary['cells']['new64']['mean_rmse'] = .1
    elif change == 'record':
        audit['results']['cells']['old16']['rows'][0]['status'] = 'incomplete'
    elif change == 'timing':
        audit['results']['cells']['new16']['median_request_ms'] = summary['cells']['new16']['median_request_ms'] = None
    else:
        audit['results']['cells']['new16']['failure_counts']['warmup'] = 1
        summary['cells']['new16']['failure_counts']['warmup'] = 1
    with pytest.raises(ValueError):
        plot.extract(audit, summary)


def test_completed_matrix_does_not_promote_incomplete_training():
    audit, summary = scalar_fixture()
    audit['scientific_status'] = 'REFERENCE_INCOMPLETE'
    audit['results']['new_fit_status'] = summary['new_fit_status'] = 'FIT_ONLY_INCOMPLETE'
    audit['results']['continuation']['reference_complete'] = summary['continuation']['reference_complete'] = False
    values = plot.extract(audit, summary)
    assert all('diagnostic' in r['label'] for r in values['cells'])
    assert values['scientific_status'] == 'REFERENCE_INCOMPLETE'


def test_fabricated_render_zero_inclusive_and_candidate_error_only(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    audit, summary = scalar_fixture()
    for row in audit['results']['cells']['old16']['rows']:
        row['rmse'] = 0.
    audit['results']['cells']['old16']['mean_rmse'] = summary['cells']['old16']['mean_rmse'] = 0.
    values = plot.extract(audit, summary)
    figures = []
    original = plt.close
    monkeypatch.setattr(plt, 'close', lambda fig=None: figures.append(fig) if hasattr(fig, 'axes') else original(fig))
    plot.figure(values, tmp_path)
    fig = figures[-1]
    assert [ax.get_xlim()[0] for ax in fig.axes] == [0., 0.]
    assert len(fig.axes[0].lines) == 1 and len(fig.axes[1].lines) == 0
    assert (tmp_path/'benchmark.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path/'benchmark.pdf').read_bytes().startswith(b'%PDF')
    original(fig)


def opaque_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, 'ROOT', tmp_path)

    def save(name, value):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value if isinstance(value, str) else json.dumps(value))
        return plot.descriptor(p)

    sources = {name: save(name, '# fabricated source') for name in plot.HELD}
    monkeypatch.setattr(plot, 'HELD', {k: v['sha256'] for k, v in sources.items()})
    registered = save('producer.py', '# qualified fake source')
    reg = save('registration.json', {})
    freeze, qual = save('freeze.json', {}), save('qualification.json', {})
    process = save('process/process.json', {'status': 'completed', 'observed_exit_code': 0})
    summary = save('study/summary.json', {'opaque': 'not extracted during admission'})
    bank = save('study/bank.npz', 'not an NPZ: decoding forbidden')
    inputs = {'registration': reg, 'freeze': freeze, 'qualification': qual, 'process': process,
              'files': {'summary.json': {k: summary[k] for k in ('bytes', 'sha256')}}}
    audit = save('audit/audit.json', {'status': 'PASS', 'agreement': True, 'study': str(tmp_path/'study'),
                                    'scientific_status': 'REFERENCE_COMPLETE', 'inputs': inputs})
    wrapper = {'state': 'EXITED', 'observed_exit_code': 0, 'success': True, 'audit_status': 'PASS', 'agreement': True,
        'error': None, 'closure_error': None, 'evidence_errors': {}, 'sources_unchanged': True, 'inputs_unchanged': True,
        'audit_output': audit, 'sources_before': sources, 'sources_after': copy.deepcopy(sources),
        'inputs_before': {'opaque_bank': bank}, 'inputs_after': {'opaque_bank': bank},
        'helper': save('wrapper.py', '# fake'), 'log': save('audit.log', 'PASS'), 'qualification': qual,
        'registration': reg, 'freeze': freeze, 'evaluation_process': process,
        'scientific_status': 'REFERENCE_COMPLETE', 'admission_before': inputs, 'admission_after': inputs,
        'command': ['.venv/bin/python', '-u', plot.AUDITOR, '--study', 'study', '--process', 'process/process.json',
                    '--freeze', 'freeze.json', '--output', 'audit/audit.json']}
    closure = save('audit-process.json', wrapper)
    monkeypatch.setattr(plot.admission, 'authenticate', lambda *args: (
        {'source_sha256': {'producer.py': registered['sha256']}}, copy.deepcopy(inputs), {},
        {'status': 'completed', 'observed_exit_code': 0}, {}, {}))
    monkeypatch.setattr(plot.admission.np, 'load', lambda *a, **k: pytest.fail('array decoder forbidden'))
    return (tmp_path/'study', Path(audit['path']), Path(closure['path'])), wrapper, save


def test_opaque_closed_admission(tmp_path, monkeypatch):
    args, _, _ = opaque_fixture(tmp_path, monkeypatch)
    paths, pins = plot.authenticate(*args)
    assert set(paths) == {'audit', 'summary'}
    assert pins['registered_source:producer.py']['path'] == str(tmp_path/'producer.py')


@pytest.mark.parametrize('change', ['live', 'output_hash', 'source', 'registered_source', 'bank', 'command', 'admission'])
def test_admission_rejects_changed_or_unclosed_evidence(tmp_path, monkeypatch, change):
    args, wrapper, save = opaque_fixture(tmp_path, monkeypatch)
    if change == 'live':
        wrapper['state'] = 'running'
    elif change == 'output_hash':
        wrapper['audit_output']['sha256'] = '0'*64
    elif change == 'source':
        save(plot.AUDITOR, '# changed')
    elif change == 'registered_source':
        save('producer.py', '# changed')
    elif change == 'bank':
        save('study/bank.npz', 'changed opaque bytes')
    elif change == 'command':
        wrapper['command'][4] = 'other-study'
    else:
        wrapper['admission_after'] = {}
    save('audit-process.json', wrapper)
    with pytest.raises(ValueError):
        plot.authenticate(*args)
