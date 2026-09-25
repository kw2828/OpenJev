"""Fabricated scalar and opaque-byte publication checks; no scientific reads."""
import copy
import csv
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_robot_position_observer as pub


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value) if isinstance(value, bytes) else path.write_text(json.dumps(value, allow_nan=False))
    return pub.descriptor(path)


def fixture():
    audit = pub.auditor
    rates = {'local_affine': .003, 'temporal_affine': .001, **dict.fromkeys(audit.parent_audit.FIXED),
             **audit.parent_audit.CACHED_RATES}
    cfg = {'inherited_rates': rates, 'all_evaluation_data_exposed': True}
    selection = {'selected_rates': {audit.CANDIDATE: .001}}
    rows, attempts = [], []
    for identities in (audit.parent_audit.identities(), audit.identities()):
        for name in audit.EXPOSED:
            for item in identities:
                fail = item['arm'] == 'observer_learned'
                common = {'recording': name, 'fit_key': item['key'], 'arm': item['arm'],
                          'seed': item['seed'], 'learning_rate': item['learning_rate']}
                error = {'type': 'FailedTrainingAttempt'} if fail else None
                for horizon in (64, 128):
                    rows.append({**common, 'horizon': horizon, 'status': 'FAILED' if fail else 'PASS', 'error': error,
                                 'metrics': None if fail else {'standardized_rmse': 1., 'standardized_sse': 22*horizon*6,
                                 'scalars': 22*horizon*6, 'windows': 22, 'horizon': horizon, 'physical_rmse_deg': 2.,
                                 'per_joint_rmse_deg': [2.]*6}})
                attempts.append({**common, 'status': 'FAILED' if fail else 'PASS', 'errors': [error, error],
                                 'prediction_file': None if fail else f"prediction-{name}-{item['key']}.npz"})
    resources = [{**r, **audit.storage(r['arm']), 'status': 'PASS', 'error': None,
                  'timing': {'seconds': [1.]*20, 'median_seconds': 1., 'p95_seconds': 1.}}
                 for r in audit.resource_identities(selection, rates)]
    conditions = [{'name': name, 'passed': i != 1} for i, name in enumerate(audit.CONDITIONS)]
    gate = {'status': 'POSITION_OBSERVER_DEVELOPMENT_FAIL', 'passed': 4, 'total': 5, 'conditions': conditions,
            'details': {name: {'means': dict.fromkeys(audit.ELIGIBLE, 1.)} for name in audit.EXPOSED},
            'equal_file_means': dict.fromkeys(audit.ELIGIBLE, 1.),
            'costs': {arm: {'latency': 1., 'bytes': 2888} for arm in audit.ELIGIBLE}}
    counts = dict(zip(('fresh_fits','fixed_models','inherited_models','final_checkpoints','diagnostic_probes',
                      'raw_gradient_banks','metric_rows','inherited_metric_rows','new_metric_rows','prediction_attempts',
                      'inherited_prediction_attempts','new_prediction_attempts','resource_rows','condition_rows',
                      'parent_forecast_replays'), (6,3,36,45,7,13,488,416,72,244,208,36,46,5,0), strict=True))
    value = {'results': {'version': audit.STUDY_VERSION, 'config': cfg, 'selection': selection, 'rows': rows, 'result': gate},
             'resources': resources, 'prediction_attempts': attempts, 'probes': [{**r, 'status': 'PASS', 'outcome': 'finite', 'gradient': {'norm64': 1., 'all_finite': True},
                        'native_norm': {'kind': 'finite', 'value': 1.}, 'prefix': {'max_state_abs': 2.}} for r in audit.probe_schedule()],
             'fit_diagnostics': {r['key']: {} for r in audit.identities()[:6]}, 'counts': counts}
    pin = {'bytes': 1, 'sha256': 'f'*64}
    common = ('normalizers.npz','linear.npz','causal_ridge_1.npz','causal_ridge_1.json','causal_ridge_100.npz',
              'causal_ridge_100.json',*(f'batches-{s}.npz' for s in (8101,8102,8103)),
              'results.json','fits.json','resources.json','prediction-attempts.json')
    inputs = {'parent/'+n: dict(pin) for n in common}
    inputs.update({'parent/'+r['key']+'/final.npz': dict(pin) for r in audit.inherited_identities(rates) if r['arm'] not in audit.REFS})
    inputs['diagnostic/observer_learned-8103-lr0/final.npz'] = dict(pin)
    inputs.update({'data/'+n: dict(pin) for n in (*audit.EXPOSED, *(f'fit{i}' for i in range(7)))})
    plan = {'config': cfg, 'sources': dict.fromkeys(audit.SOURCES, pin), 'inputs': inputs}
    # Independent literal inventory construction, not publication's helper.
    names = {'registration.json','runtime.json','fits.json','checkpoint-barrier.json','parameter-checks.json',
             'prediction-attempts.json','resources.json','results.json','probes.json'}
    names |= {'sources/'+n for n in audit.SOURCES}
    names |= {n for n in inputs if not n.startswith('data/')}
    names |= {r['key']+'/final.npz' for r in (*audit.identities(), *audit.inherited_identities(rates)) if r['arm'] not in audit.REFS}
    for r in audit.identities()[:6]:
        names |= {r['key']+'/'+n for n in ('initial.npz','optimizer.npz','trace.json','fit-receipt.json',
                                         'diagnostics.json','diagnostic-summary.json','last-gradient.npz')}
    names |= {f'completed-fit-{i:02d}.json' for i in range(1,7)}
    names |= {'probes/'+r['key']+'/'+n for r in audit.probe_schedule() for n in ('receipt.json','preclip.npz')}
    names |= {f'completed-prediction-{i:03d}.json' for i in range(1,37)}
    names |= {f'completed-timing-{i:02d}.json' for i in range(1,47)}
    names |= {'dev-windows-'+n+'.npz' for n in audit.EXPOSED}
    names |= {r['prediction_file'] for r in attempts[208:] if r['prediction_file'] is not None}
    inventory = dict.fromkeys(names, pin)
    return value, plan, inventory


def test_full_negative_roster_preserves_parent_failure_and_only_four_exclusions():
    value, plan, inventory = fixture()
    assert len(inventory) == 342 and len(value['results']['rows']) == 488
    assert len(value['prediction_attempts']) == 244 and len(value['resources']) == 46
    assert len(pub.FAMILIES) == len(pub.LABELS) == 19
    pub.validate_scalars(value)
    excluded = pub.study_roster(inventory, plan, value['prediction_attempts'])
    assert set(excluded) == {'dev-windows-'+n+'.npz' for n in pub.auditor.EXPOSED}
    assert all(r['status'] == 'FAILED' for r in value['results']['rows'] if r['arm'] == 'observer_learned')


@pytest.mark.parametrize('damage', ['missing_probe','missing_fit_diagnostic','old_prediction','measurement','traversal','lost_parent_row','promoted'])
def test_omissions_extra_banks_and_promotion_reject(damage):
    value, plan, inventory = fixture()
    if damage == 'missing_probe': inventory.pop('probes/identity-8101-batch0/preclip.npz')
    elif damage == 'missing_fit_diagnostic': inventory.pop('observer_position-8101-lr0/diagnostic-summary.json')
    elif damage == 'old_prediction': inventory[value['prediction_attempts'][0]['prediction_file']] = {'bytes':1,'sha256':'f'*64}
    elif damage == 'measurement': inventory['raw_data/measured.mat'] = {'bytes':1,'sha256':'f'*64}
    elif damage == 'traversal': inventory['../outside'] = {'bytes':1,'sha256':'f'*64}
    elif damage == 'lost_parent_row': value['results']['rows'].pop(0)
    else: value['results']['result']['status'] = 'POSITION_OBSERVER_DEVELOPMENT_PASS'
    with pytest.raises(ValueError):
        pub.validate_scalars(value); pub.study_roster(inventory, plan, value['prediction_attempts'])


def test_absent_new_forecast_is_retained_as_failure_not_success():
    value, plan, inventory = fixture(); row = value['prediction_attempts'][208]
    inventory.pop(row['prediction_file']); row.update(prediction_file=None, status='FAILED', errors=[{'type':'Known'}, {'type':'Known'}])
    assert len(inventory) == 341
    pub.study_roster(inventory, plan, value['prediction_attempts'])
    row['status'] = 'PASS'
    with pytest.raises(ValueError): pub.study_roster(inventory, plan, value['prediction_attempts'])


def test_parent_rates_are_fixed_and_failed_observer_never_selected():
    value, _, _ = fixture(); selection = value['results']['selection']; rates = value['results']['config']['inherited_rates']
    for row in value['results']['rows']:
        answer = pub.selected(row, selection, rates)
        if row['arm'] == 'observer_learned': assert answer is False
        if row['arm'] == 'local_affine': assert answer is (row['learning_rate'] == .003)
        if row['arm'] == 'observer_position': assert answer is (row['learning_rate'] == .001)


def test_tables_retain_every_row_and_missing_cost_is_never_zero(tmp_path):
    value, _, _ = fixture(); row = value['resources'][0]
    row.update(status='FAILED', error={'type':'Known'}, timing=None)
    pub.write_tables(tmp_path, value)
    with (tmp_path/'scores.csv').open() as f: scores = list(csv.DictReader(f))
    with (tmp_path/'costs.csv').open() as f: costs = list(csv.DictReader(f))
    assert len(scores) == 488 and len(costs) == 46
    assert costs[0]['median_seconds'] == costs[0]['p95_seconds'] == ''
    assert costs[0]['persistent_bytes'] == '2888'
    assert all(r['standardized_rmse'] == '' for r in scores if r['status'] == 'FAILED')
    text = (tmp_path/'table.md').read_text()
    assert 'no complete parent recipe' in text and 'POSITION_OBSERVER_DEVELOPMENT_FAIL' in text
    assert json.loads((tmp_path/'audited-summary.json').read_text())['probes'] == value['probes']


@pytest.mark.parametrize('name', ['../x', '/absolute', 'a/../x', '.', ''])
def test_unsafe_public_paths_fail(name, tmp_path):
    with pytest.raises(ValueError): pub.safe_path(tmp_path, name)


def test_render_full_and_detail_keep_every_outlier_and_shared_axes(tmp_path, monkeypatch):
    value, _, _ = fixture()
    # One selected and one unselected outlier in each file, plus an outlying mean.
    for name in pub.auditor.EXPOSED:
        instances = [r for r in value['results']['rows'] if r['recording'] == name
                     and r['arm'] == 'local_affine' and r['horizon'] == 128]
        instances[0]['metrics']['standardized_rmse'] = 3.
        instances[1]['metrics']['standardized_rmse'] = 1e8
        value['results']['result']['details'][name]['means']['local_affine'] = 4.
    # Keep an explicit failed point visible in both views.
    next(r for r in value['results']['rows'] if r['arm'] == pub.auditor.CANDIDATE and r['learning_rate'] == .001
         and r['horizon'] == 128).update(status='FAILED', metrics=None, error={'type': 'KnownNumeric'})
    class Axis:
        def __init__(self):
            self.limits = None; self.scale = None; self.scale_args = {}; self.points = []
            self.lines = []; self.annotations = []; self.texts = []; self.label = ''; self.title = ''
        def scatter(self, x, y, **kw): self.points.append((x, y, kw))
        def plot(self, *args, **kw): self.lines.append((args, kw))
        def annotate(self, text, position, **kw): self.annotations.append((text, position, kw))
        def text(self, *args, **kw): self.texts.append(args)
        def get_xaxis_transform(self): return None
        def set_title(self, text, **kw): self.title = text
        def set_ylabel(self, label): self.label = label
        def set_ylim(self, *args): self.limits = args
        def set_yscale(self, scale, **kw): self.scale = scale; self.scale_args = kw
        def set_xticks(self, *args, **kw): pass
        def grid(self, *args, **kw): pass
        def set_axisbelow(self, *args): pass
    class Axes:
        def __init__(self): self.flat = [Axis() for _ in range(6)]
        def __getitem__(self, item): return self.flat[item[0]*2+item[1]]
    class Figure:
        def __init__(self): self.texts = []; self.paths = []
        def suptitle(self, text, **kw): self.title = text
        def text(self, *args, **kw): self.texts.append(args[2])
        def tight_layout(self, *args, **kw): pass
        def savefig(self, path, **kw): self.paths.append(path.name); path.write_bytes(b'fabricated figure')
    figures = []
    def subplots(*args, **kw):
        fig, axes = Figure(), Axes(); figures.append((fig, axes)); return fig, axes
    mpl = types.ModuleType('matplotlib'); mpl.use = lambda *args: None
    pyplot = types.ModuleType('matplotlib.pyplot'); pyplot.subplots = subplots; pyplot.close = lambda *args: None
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl); monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    untouched = copy.deepcopy(value)
    pub.render(tmp_path, value)
    assert value == untouched and len(figures) == 2
    (full_fig, full), (detail_fig, detail) = figures
    for axes in (full, detail):
        assert len({a.limits for a in axes.flat[:4]}) == 1 and axes.flat[0].limits[0] == 0
        assert [len(a.points) for a in axes.flat] == [54, 55, 55, 55, 46, 46]
        assert sum(point[2]['marker'] == 'x' for axis in axes.flat[:4] for point in axis.points) == 36
        assert all(a.scale == 'log' and 'log scale' in a.label for a in axes.flat[4:])
        assert any(text[2] == 'failed' for text in axes.flat[0].texts)
    assert all(a.scale == 'symlog' and a.scale_args == {'linthresh': .1} for a in full.flat[:4])
    assert all('symlog' in a.label and a.limits[1] > 1e8 for a in full.flat[:4])
    assert all(a.limits == (0, 2.) for a in detail.flat[:4])
    for full_axis, detail_axis in zip(full.flat[:4], detail.flat[:4], strict=True):
        originals = [v for _, v, _ in full_axis.points]
        shown = [v for _, v, _ in detail_axis.points]
        assert shown == [min(v, 2.) for v in originals]
        arrows = [r for r in detail_axis.annotations if r[0] == '↑']
        assert len(arrows) == sum(v > 2. for v in originals)
        assert all(position[1] == 2. and opts['annotation_clip'] is False for _, position, opts in arrows)
        assert len([r for r in detail_axis.annotations if r[0] == 'mean ↑']) == 1
        assert any(args[1] == [2., 2.] and opts['linestyle'] == '--' and opts['clip_on'] is False
                   for args, opts in detail_axis.lines)
        assert all(opts['clip_on'] is False for _, _, opts in detail_axis.points)
        assert not full_axis.annotations
    assert full_fig.paths == ['benchmark.png', 'benchmark.svg']
    assert detail_fig.paths == ['benchmark-detail.png', 'benchmark-detail.svg']
    assert 'FULL RANGE' in full_fig.title and 'DETAIL 0–2' in detail_fig.title
    assert '4 of 5 criteria pass (all 5 required)' in full_fig.title
    assert 'See benchmark.png/SVG for the complete range' in detail_fig.texts[0]
    assert 'linear from 0 to 0.1' in full_fig.texts[0]


def opaque(path, value=b'opaque derived evidence, not a valid NPZ'):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    return pub.descriptor(path)


def pin(path):
    return {'path': str(path.resolve()), **pub.descriptor(path)}


def admitted_tree(root, monkeypatch):
    value, plan, inventory = fixture(); monkeypatch.setattr(pub, 'ROOT', root)
    study, audit, eng = root/pub.STUDY, root/pub.AUDIT, root/pub.ENGINEERING
    for name in inventory: opaque(study/name)
    plan['qualification'] = {'path': str(eng/'qualification-01/receipt.json'), 'sha256': 'b'*64, 'bytes': 1}
    for name in ('parent_audit', 'parent_publication_manifest'):
        path = root/'old'/name; write(path, {'opaque': name}); plan[name] = pin(path)
    for name in (*pub.DELIVERY_SOURCES, pub.LAUNCHER): opaque(root/name)
    source = root/'scripts/audit_robot_position_observer.py'; opaque(source)
    registration = root/'research/robot-position-observer-registration.json'; write(registration, plan)
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log'): opaque(eng/name)
    for name, payload in (('registration.json', plan), ('results.json', value['results']), ('resources.json', value['resources']),
                          ('prediction-attempts.json', value['prediction_attempts']), ('probes.json', value['probes'])):
        write(study/name, payload)
    for key, summary in value['fit_diagnostics'].items(): write(study/key/'diagnostic-summary.json', summary)
    inventory = {name: pub.descriptor(study/name) for name in inventory}
    write(study/'manifest.json', {'files': inventory}); write(study/'receipt.json', {'status': 'PASS'})
    inputs = {'registration': pin(registration), 'manifest': pin(study/'manifest.json')}
    write(eng/'registration-preflight-01.json', {'status': 'PASS', 'registration': inputs['registration'],
          'qualification': plan['qualification'], 'sources': 54, 'inputs': 61, 'model_calls': 0,
          'numeric_decodes': 0, 'optimizer_updates': 0, 'scope': 'metadata only'})
    value.update(status='PASS', agreement=True, study=str(study), registration_sha256=inputs['registration']['sha256'],
                 source_pins=plan['sources'], inputs=inputs, auditor=pin(source))
    write(audit, value); write(audit.parent/'manifest.json', {'files': {'audit.json': pub.descriptor(audit)}})
    launch = {'command': ['.venv/bin/python', 'scripts/audit_robot_position_observer.py', '--study', pub.STUDY,
                         '--run-receipt', pub.ENGINEERING+'/run-process-01.json', '--output', pub.AUDIT],
              'started_utc': '2026-01-01T00:00:00+00:00', 'thread_env': dict.fromkeys(pub.auditor.THREADS, '1'), 'cap_seconds': 180}
    write(eng/'audit-launch-01.json', launch); log = opaque(eng/'audit-process-01.log')
    write(eng/'audit-process-01.json', {**launch, 'returncode': 0, 'external_timeout': False, 'elapsed_seconds': 2., 'log': log})
    closure = {'status': 'PASS', 'source_matches_registration': True, 'scope': 'original closure'}
    for key, path in {'auditor': source, 'audit_output': audit, 'audit_manifest': audit.parent/'manifest.json',
                      'process': eng/'audit-process-01.json', 'launch': eng/'audit-launch-01.json',
                      'log': eng/'audit-process-01.log', 'registration': registration}.items():
        closure[key] = pin(path)
    write(eng/'audit-closure-01.json', closure)
    monkeypatch.setattr(pub.auditor, 'authenticate', lambda *args: (copy.deepcopy(plan), copy.deepcopy(inputs)))
    monkeypatch.setattr(pub, 'qualification_files', lambda *args: {})
    monkeypatch.setattr(pub, 'publication_qualification_files', dict)
    return study, audit, eng, plan, value


def test_closed_metadata_and_original_audit_bytes_join_before_saved_numbers(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    auth = pub.authenticate(study, audit, eng)
    assert len(auth['excluded']) == 4 and len(auth['inventory']) == 342
    assert auth['audit']['results']['result']['status'] == 'POSITION_OBSERVER_DEVELOPMENT_FAIL'


@pytest.mark.parametrize('damage', ['exit', 'timeout', 'overcap', 'command', 'log', 'closure', 'manifest',
                                   'source', 'results', 'probe', 'fit_diagnostic', 'inputs', 'agreement', 'preflight'])
def test_original_audit_and_scalar_mismatches_fail_closed(tmp_path, monkeypatch, damage):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    if damage in ('exit', 'timeout', 'overcap', 'command'):
        path = eng/'audit-process-01.json'; data = pub.read(path)
        if damage == 'exit': data['returncode'] = 1
        elif damage == 'timeout': data['external_timeout'] = True
        elif damage == 'overcap': data['elapsed_seconds'] = 181.
        else: data['command'][1] = 'unrelated.py'
        write(path, data)
    elif damage == 'log': opaque(eng/'audit-process-01.log', b'changed')
    elif damage == 'closure': write(eng/'audit-closure-01.json', {'status': 'FAIL'})
    elif damage == 'manifest': write(audit.parent/'manifest.json', {'files': {}})
    elif damage == 'source': opaque(tmp_path/'scripts/audit_robot_position_observer.py', b'changed')
    elif damage == 'results':
        data = pub.read(study/'results.json'); data['result']['passed'] = 5; write(study/'results.json', data)
    elif damage == 'probe': write(study/'probes.json', [])
    elif damage == 'fit_diagnostic': write(study/'observer_position-8101-lr0/diagnostic-summary.json', {'changed': True})
    elif damage == 'preflight':
        path = eng/'registration-preflight-01.json'; data = pub.read(path); data['inputs'] = 60; write(path, data)
    else:
        data = pub.read(audit)
        if damage == 'inputs': data['inputs']['manifest']['sha256'] = 'c'*64
        else: data['agreement'] = False
        write(audit, data); write(audit.parent/'manifest.json', {'files': {'audit.json': pub.descriptor(audit)}})
        closure = pub.read(eng/'audit-closure-01.json'); closure['audit_output'] = pin(audit)
        closure['audit_manifest'] = pin(audit.parent/'manifest.json'); write(eng/'audit-closure-01.json', closure)
    with pytest.raises(ValueError): pub.authenticate(study, audit, eng)


def test_opaque_package_roundtrip_and_measurement_exclusions(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    for function in ('render', 'render_probes'):
        monkeypatch.setattr(pub, function, lambda *args: None)
    receipt = pub.publish(study, audit, eng, tmp_path/pub.OUTPUT)
    output = tmp_path/pub.OUTPUT; manifest = pub.read(output/'manifest.json')
    assert receipt['status'] == 'PASS' and receipt['scientific_status'] == 'POSITION_OBSERVER_DEVELOPMENT_FAIL'
    assert receipt['array_decodes'] == receipt['model_calls'] == receipt['rescoring_calls'] == 0
    assert len(manifest['excluded_measurement_payloads']) == 4 and len(manifest['external_measurement_inputs']) == 11
    assert not any((output/'study'/n).exists() for n in manifest['excluded_measurement_payloads'])
    for name, expected in manifest['files'].items(): assert pub.descriptor(output/name) == expected
    for name, source in manifest['copy_sources'].items(): assert pub.descriptor(output/name) == pub.descriptor(source['path'])


def test_changed_source_during_copy_prevents_success_receipt(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    def change(*args): opaque(study/'runtime.json', b'changed after admission')
    monkeypatch.setattr(pub, 'render', change); monkeypatch.setattr(pub, 'render_probes', lambda *args: None)
    with pytest.raises(ValueError): pub.publish(study, audit, eng, tmp_path/pub.OUTPUT)
    assert not (tmp_path/pub.OUTPUT/'receipt.json').exists()


def scientific_qualifications(root, plan, statuses=('FAILED', 'PASS')):
    files = []
    commands = [['.venv/bin/ruff', 'check', *[n for n in pub.auditor.QUALIFICATION_SOURCES if n.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_position_observer.py',
                 'tests/test_robot_position_observer_study.py', 'tests/test_audit_robot_position_observer.py']]
    for i, status in enumerate(statuses, 1):
        folder = root/f'qualification-{i:02d}'
        sources = {n: write(folder/'sources'/n, f'{i}:{n}'.encode()) for n in pub.auditor.SOURCES}
        launcher = write(folder/'sources'/pub.LAUNCHER, f'launcher{i}'.encode())
        launch = {'created_utc': '2026-01-01T00:00:00+00:00', 'sources': sources, 'launcher': launcher, 'commands': [],
                  'thread_env': dict.fromkeys(pub.auditor.THREADS, '1'), 'command_cap_seconds': 180, 'scope': 'fabricated'}
        write(folder/'launch.json', launch); write(folder/'qualification-helper.py', b'original helper')
        rows = []
        for j, cmd in enumerate(commands, 1):
            log = folder/f'command-{j:02d}.log'; desc = write(log, f'{status}{j}'.encode())
            row = {'command': cmd, 'returncode': 1 if status == 'FAILED' and j == 2 else 0,
                   'seconds': .1, 'log': str(log), 'sha256': desc['sha256']}
            rows.append(row); write(folder/f'process-{j:02d}.json', row)
        receipt = {**launch, 'commands': rows, 'status': status, 'sources_unchanged': True}
        write(folder/'receipt.json', receipt); files.append(folder)
        plan.update(sources=sources, launcher=launcher, qualification=pin(folder/'receipt.json'))
    return files


def test_all_original_science_qualifications_preserved(tmp_path):
    _, plan, _ = fixture(); folders = scientific_qualifications(tmp_path, plan)
    files = pub.qualification_files(tmp_path, plan)
    assert len(files) == 124  # 54 source files, launcher, seven original receipt/log/helper files, per attempt.
    assert files['engineering/qualification-01/receipt.json'] == folders[0]/'receipt.json'
    assert pub.read(folders[0]/'receipt.json')['status'] == 'FAILED'
    assert pub.read(folders[1]/'receipt.json')['status'] == 'PASS'


@pytest.mark.parametrize('damage', ['gap', 'snapshot', 'log', 'process', 'command', 'launcher', 'unlisted'])
def test_science_qualification_admission(damage, tmp_path):
    _, plan, _ = fixture(); folders = scientific_qualifications(tmp_path, plan)
    f = folders[0]
    if damage == 'gap': f.rename(tmp_path/'qualification-03')
    elif damage == 'snapshot': (f/'sources'/pub.auditor.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'log': (f/'command-01.log').write_bytes(b'changed')
    elif damage == 'process': write(f/'process-01.json', {})
    elif damage == 'launcher': (f/'sources'/pub.LAUNCHER).write_bytes(b'changed')
    elif damage == 'unlisted': write(f/'surprise.npz', b'not a measurement')
    else:
        r = pub.read(f/'receipt.json'); r['commands'][0]['command'] = ['echo', 'PASS']; write(f/'receipt.json', r)
        write(f/'process-01.json', r['commands'][0])
    with pytest.raises(ValueError): pub.qualification_files(tmp_path, plan)


def publication_qualifications(root, statuses=('FAIL', 'PASS')):
    commands = [['.venv/bin/ruff', 'check', *pub.DELIVERY_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', pub.DELIVERY_SOURCES[1]]]
    folders = []
    for i, status in enumerate(statuses, 1):
        folder = root/pub.PUBLICATION_ENGINEERING/f'qualification-{i:02d}'
        sources = {n: write(folder/'sources'/n, f'publication{i}:{n}'.encode()) for n in pub.DELIVERY_SOURCES}
        for n in pub.DELIVERY_SOURCES: write(root/n, (folder/'sources'/n).read_bytes())
        pre = {'sources': sources, 'commands': commands, 'thread_env': dict.fromkeys(pub.auditor.THREADS, '1')}
        rows = []
        for j, cmd in enumerate(commands, 1):
            log = folder/f'command-{j:02d}.log'; d = write(log, f'{status}:{j}'.encode())
            rows.append({'command': cmd, 'returncode': 1 if status == 'FAIL' and j == 2 else 0, 'seconds': .1,
                         'log': str(log), 'sha256': d['sha256']})
        write(folder/'preflight.json', pre); write(folder/'receipt.json', {**pre, 'commands': rows, 'status': status, 'sources_unchanged': True})
        folders.append(folder)
    return folders


def test_publication_qualification_current_source_and_failed_history(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, 'ROOT', tmp_path)
    folders = publication_qualifications(tmp_path)
    files = pub.publication_qualification_files()
    assert len(files) == 12 and pub.read(folders[0]/'receipt.json')['status'] == 'FAIL'
    (tmp_path/pub.DELIVERY_SOURCES[0]).write_bytes(b'changed')
    with pytest.raises(ValueError): pub.publication_qualification_files()


def test_publication_self_consistent_unrelated_commands_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, 'ROOT', tmp_path)
    folder = publication_qualifications(tmp_path, ('PASS',))[0]
    pre, r = pub.read(folder/'preflight.json'), pub.read(folder/'receipt.json')
    pre['commands'][0] = ['echo', 'pass']; r['commands'][0]['command'] = ['echo', 'pass']
    write(folder/'preflight.json', pre); write(folder/'receipt.json', r)
    with pytest.raises(ValueError): pub.publication_qualification_files()


def test_seven_probe_chart_preserves_native_overflow_and_missing_diagnostics(tmp_path, monkeypatch):
    value, _, _ = fixture()
    value['probes'][0].update(gradient={'norm64': 1e20, 'all_finite': True},
                              native_norm={'kind': 'positive_infinity', 'value': None})
    value['probes'][1].update(gradient=None, native_norm=None, prefix={})
    value['probes'][2]['gradient']['norm64'] = 0.
    class Axis:
        def __init__(self): self.points = []; self.texts = []; self.scale = None; self.ticks = None
        def scatter(self, x, y, **kw): self.points.append((x,y,kw))
        def text(self, *args, **kw): self.texts.append(args)
        def get_xaxis_transform(self): return None
        def set_yscale(self, scale, **kw): self.scale = scale
        def set_title(self, *args): pass
        def set_xticks(self, *args, **kw): self.ticks = args
        def grid(self, *args, **kw): pass
        def set_axisbelow(self, *args): pass
    class Figure:
        def __init__(self): self.paths = []; self.caption = ''
        def suptitle(self, *args, **kw): pass
        def text(self, *args, **kw): self.caption = args[2]
        def tight_layout(self, **kw): pass
        def savefig(self, path, **kw): self.paths.append(path.name)
    fig, axes = Figure(), [Axis(),Axis()]
    mpl = types.ModuleType('matplotlib'); mpl.use = lambda *args: None
    pyplot = types.ModuleType('matplotlib.pyplot'); pyplot.subplots = lambda *args, **kw: (fig,axes); pyplot.close = lambda *args: None
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl); monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    before = copy.deepcopy(value); pub.render_probes(tmp_path, value)
    assert value == before and all(a.scale == 'log' and len(a.ticks[1]) == 7 for a in axes)
    assert axes[0].points[0][1] == 1e20 and axes[0].points[0][2]['marker'] == 'X'
    assert len(axes[0].points) == 5 and len(axes[1].points) == 6
    assert any('positive_infinity' in r[2] for r in axes[0].texts)
    assert any(r[2] == 'zero' for r in axes[0].texts)
    assert any('not captured' in r[2] for r in axes[0].texts)
    assert fig.paths == ['diagnostic-probes.png','diagnostic-probes.svg']
    assert 'zero optimizer steps' in fig.caption and 'do not decide the five' in fig.caption
