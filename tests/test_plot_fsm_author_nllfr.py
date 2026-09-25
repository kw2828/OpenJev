"""Fabricated scalar/opaque-file tests; no measured inputs or model calls."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('nllfr_plot', ROOT/'scripts/plot_fsm_author_nllfr.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def scalar_fixture():
    ref = plot.bla_plot.reference_plot
    names = sorted(ref.RECORDS)

    def records(value):
        return [{'record_id': name, 'status': 'complete', 'rmse': value, 'requests': 32, 'horizon': 128}
                for name in names]

    old = {'status': 'DEVELOPMENT_PASS', 'candidate': ref.CANDIDATE, 'strongest_control': ref.LINEAR[0],
           'conditions': dict.fromkeys(ref.CONDITIONS, True), 'passed': 9, 'total': 9, 'families': {}}
    evaluations = []
    for family in ref.FAMILIES:
        seeded = family in ref.SEEDED
        for i, seed in enumerate(ref.SEEDS if seeded else (None,)):
            evaluations.append({'family': family, 'seed': seed, 'rows': records(i+1),
                'request_ms': [i+1.]*24, 'median_request_ms': i+1., 'timing_error': None,
                'persistent_numeric_bytes': 100})
        old['families'][family] = {'eligible': True, 'score_eligible': True, 'latency_eligible': True,
            'storage_eligible': True, 'mean_rmse': 2. if seeded else 1.,
            'mean_seed_median_ms': 2. if seeded else 1., 'persistent_numeric_bytes': 100}
    bla_eval = {'rows': records(.4), 'request_ms': [3.]*24, 'median_request_ms': 3., 'timing_error': None}
    bla = {'status': 'REFERENCE_COMPLETE', 'mean_rmse': .4, 'median_request_ms': 3.,
           'persistent_numeric_bytes': 200, 'fit': {'status': 'complete'}}
    evaluation = {'rows': [dict(row, rmse=float(i)) for i, row in enumerate(records(0.))],
                  'timings': [{'record_id': name, 'start': start, 'request_ms': float(i*2+j+1)}
                              for i, name in enumerate(names) for j, start in enumerate((0, 7936))],
                  'median_request_ms': 12.5, 'timing_error': None, 'persistent_numeric_bytes': 300}
    summary = {'status': 'REFERENCE_INCOMPLETE', 'fit_status': 'iteration_cap_reached',
               'mean_rmse': 5.5, 'median_request_ms': 12.5, 'persistent_numeric_bytes': 300}
    return summary, evaluation, bla, bla_eval, old, evaluations


def test_five_families_hand_means_seeds_and_incomplete_fit():
    values = plot.extract(*scalar_fixture())
    assert [r['family'] for r in values['families']] == ['author_nllfr28', 'author_bla28', *plot.bla_plot.REFERENCE_FAMILIES]
    assert [r['mean_rmse'] for r in values['families']] == [5.5, .4, 1., 2., 2.]
    assert [r['mean_seed_median_ms'] for r in values['families']] == [12.5, 3., 1., 2., 2.]
    assert [m['rmse'] for m in values['families'][-1]['members']] == [1., 2., 3.]
    assert values['scientific_status'] == 'REFERENCE_INCOMPLETE'
    assert 'FIT INCOMPLETE' in values['families'][0]['label']


@pytest.mark.parametrize('change', ['promote', 'mean', 'median', 'duplicate_record', 'duplicate_timing', 'unknown_record'])
def test_scalar_corruption_fails(change):
    args = scalar_fixture()
    if change == 'promote':
        args[0]['status'] = 'REFERENCE_COMPLETE'
    elif change == 'mean':
        args[0]['mean_rmse'] += .1
    elif change == 'median':
        args[0]['median_request_ms'] = args[1]['median_request_ms'] = 11.
    elif change == 'duplicate_record':
        args[1]['rows'][1] = args[1]['rows'][0]
    elif change == 'duplicate_timing':
        args[1]['timings'][1] = args[1]['timings'][0]
    else:
        args[1]['rows'][0]['record_id'] = '300mV-hidden'
    with pytest.raises(ValueError):
        plot.extract(*args)


def test_missing_forecast_and_timing_remain_visible():
    args = scalar_fixture()
    row = args[1]['rows'][0]
    args[1]['rows'][0] = {'record_id': row['record_id'], 'status': 'incomplete',
                        'completed_requests': 31, 'expected_requests': 32}
    args[0]['mean_rmse'] = None
    args[0]['median_request_ms'] = args[1]['median_request_ms'] = None
    args[1]['timing_error'] = 'retained numerical failure'
    args[1]['timings'] = args[1]['timings'][:3]
    first = plot.extract(*args)['families'][0]
    assert first['complete_records'] == 11 and first['mean_rmse'] is None
    assert first['mean_seed_median_ms'] is None and first['timing_error']


def test_audited_scalar_join_and_zero_error():
    args = scalar_fixture()
    for row in args[1]['rows']:
        row['rmse'] = 0.
    args[0]['mean_rmse'] = 0.
    values = plot.extract(*args)
    result = {'reference_status': 'REFERENCE_INCOMPLETE', 'fit_status': 'iteration_cap_reached',
              'reference_mean_rmse': 0., 'median_request_ms': 12.5, 'rows': args[1]['rows']}
    audit = {'scientific_status': 'REFERENCE_INCOMPLETE', 'results': result}
    plot.check_audited_scalars(values, audit)
    assert values['saved_audit_result'] == result
    result['reference_status'] = 'REFERENCE_COMPLETE'
    with pytest.raises(ValueError, match='status'):
        plot.check_audited_scalars(values, audit)


def admission_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, 'ROOT', tmp_path)
    def save(name, data):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return plot.descriptor(path)
    source = {name: save(name, '# fabricated source') for name in (plot.AUDITOR, plot.AUDITOR_TEST)}
    hashes = {k: v['sha256'] for k, v in source.items()}
    monkeypatch.setattr(plot, 'AUDITOR_SHA', hashes)
    registered = save('producer.py', '# fabricated frozen source')
    reg = save('reg.json', {'source_sha256': {'producer.py': registered['sha256']}})
    snapshots = {k: save('snapshots/'+k, '# fabricated source') for k in source}
    preflight = save('preflight.json', {'sources': source, 'snapshots': snapshots, 'commands': [['lint'], ['tests']]})
    logs = [save(f'qualification-{i}.log', 'PASS') for i in range(2)]
    qual = save('qualification.json', {'status': 'PASS', 'sources_unchanged': True,
        'sources_before': source, 'sources_after': source, 'preflight': preflight,
        'commands': [{'command': command, 'returncode': 0, 'log': log}
                     for command, log in zip((['lint'], ['tests']), logs, strict=True)]})
    save(plot.AUDITOR_FREEZE, {'status': 'FROZEN_BEFORE_EMPIRICAL_AUDIT', 'source_sha256': hashes,
        'qualification_sha256': qual['sha256'], 'qualification_runtime_preflight_sha256': preflight['sha256'],
        'registration_sha256': reg['sha256']})
    study, evaluation, reference, bla = [tmp_path/n for n in ('fit', 'evaluation', 'reference', 'bla')]
    scalar_names = ('summary.json', 'evaluation.json')
    for folder in (evaluation, bla):
        for name in scalar_names:
            save(str(folder.relative_to(tmp_path)/name), {'opaque': 'fixture scalar, never extracted'})
    for name in ('summary.json', 'evaluations.json'):
        save('reference/'+name, {})
    save('fit/final.npz', 'invalid NPZ: decoding forbidden')
    def files(folder):
        return {p.name: {k: v for k, v in plot.descriptor(p).items() if k != 'path'} for p in folder.iterdir()}
    fit_process = save('fit-process/process.json', {'status': 'completed'})
    eval_process = save('eval-process/process.json', {'status': 'completed'})
    bla_audit = save('bla-audit.json', {'inputs': {'files': files(bla)}})
    admitted = {'files': files(study), 'evaluation_files': files(evaluation),
        'evaluation_study': str(evaluation), 'source': source[plot.AUDITOR], 'registration': reg,
        'process': fit_process, 'evaluation_process': eval_process,
        'reference_summary': plot.descriptor(reference/'summary.json'),
        'reference_evaluations': plot.descriptor(reference/'evaluations.json')}
    audit = save('audit/audit.json', {'status': 'PASS', 'agreement': True, 'study': str(study),
                                    'scientific_status': 'REFERENCE_INCOMPLETE', 'inputs': admitted})
    closure = {'state': 'EXITED', 'observed_exit_code': 0, 'success': True, 'error': None, 'closure_error': None,
        'sources_unchanged': True, 'inputs_unchanged': True, 'elapsed_seconds': 1., 'timeout_seconds': 3600,
        'sources_before': source, 'sources_after': source, 'inputs_before': source, 'inputs_after': source,
        'audit_output': audit, 'helper': save('wrapper.py', '# fixture'), 'log': save('audit.log', 'PASS'),
        'qualification': qual, 'preflight': preflight, 'registration': reg, 'freeze': save('freeze.json', {}),
        'fit_process': fit_process, 'evaluation_process': eval_process,
        'audit_status': 'PASS', 'agreement': True, 'scientific_status': 'REFERENCE_INCOMPLETE',
        'command': ['.venv/bin/python', '-u', plot.AUDITOR, '--study', str(study), '--process', fit_process['path'],
                    '--evaluation-process', eval_process['path'], '--output', audit['path']]}
    process = save('audit-process.json', closure)
    terminal = {'status': 'completed', 'observed_exit_code': 0}
    monkeypatch.setattr(plot.admission, 'authenticate', lambda *args: (
        {'source_sha256': {'producer.py': registered['sha256']}}, copy.deepcopy(admitted),
        {'bla_summary': bla/'summary.json', 'bla_audit': Path(bla_audit['path'])}, terminal, terminal))
    monkeypatch.setattr(plot.admission.np, 'load', lambda *a, **k: pytest.fail('array decode forbidden'))
    args = (study, Path(audit['path']), Path(process['path']), evaluation, reference, bla)
    return args, closure, save


def test_opaque_admission_checks_without_arrays(tmp_path, monkeypatch):
    args, _, _ = admission_fixture(tmp_path, monkeypatch)
    paths, pins = plot.authenticate(*args)
    assert len(paths) == 6
    assert pins['registered_source:producer.py']['path'] == str(tmp_path/'producer.py')
    assert 'fit:final.npz' in pins


@pytest.mark.parametrize('change', ['live', 'audit_hash', 'audit_source', 'registered_source', 'qualification',
                                   'fit_process', 'argv', 'scalar', 'published_freeze'])
def test_admission_tampering_fails_before_extract(tmp_path, monkeypatch, change):
    args, closure, save = admission_fixture(tmp_path, monkeypatch)
    if change == 'live':
        closure['state'] = 'running'
    elif change == 'audit_hash':
        closure['audit_output']['sha256'] = '0'*64
    elif change == 'audit_source':
        save(plot.AUDITOR, '# tampered')
    elif change == 'registered_source':
        save('producer.py', '# tampered')
    elif change == 'qualification':
        save('qualification.json', {'status': 'FAIL'})
    elif change == 'fit_process':
        save('fit-process/process.json', {'status': 'running'})
    elif change == 'argv':
        closure['command'][4] = str(tmp_path/'other-study')
    elif change == 'scalar':
        save('evaluation/summary.json', {'changed': True})
    else:
        save(plot.AUDITOR_FREEZE, {})
    save('audit-process.json', closure)
    with pytest.raises(ValueError):
        plot.authenticate(*args)


def test_render_fabricated_only_and_explicit_incomplete_title(tmp_path):
    values = plot.extract(*scalar_fixture())
    plot.figure(values, tmp_path)
    assert (tmp_path/'benchmark.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path/'benchmark.pdf').read_bytes().startswith(b'%PDF')


def test_run_rejects_existing_output_before_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, 'authenticate', lambda *args: pytest.fail('must reject before admission'))
    with pytest.raises(ValueError, match='exclusive'):
        plot.run(*([tmp_path]*7))
