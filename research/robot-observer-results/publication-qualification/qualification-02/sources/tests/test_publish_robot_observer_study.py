"""Fabricated saved scalars and opaque bytes only; never a model or NPZ reader."""
import copy
import csv
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_robot_observer_study as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return p.descriptor(path)


def pin(path):
    return {'path': str(path.resolve()), **p.descriptor(path)}


def fixture():
    learned = ('local_affine', 'temporal_affine', 'observer_learned')
    fixed = ('last_two', 'observer_fixed', 'observer_zero')
    cached = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
    rates = dict(zip(cached, (.003, .003, .001, .003, .003, .001, .003), strict=True))
    identities = [{'key': f'{a}-{s}-lr{i}', 'arm': a, 'seed': s, 'learning_rate': r, 'origin': 'fresh'}
                  for s in (8101, 8102, 8103) for i, r in enumerate((.001, .003)) for a in learned]
    identities += [{'key': f'{a}-{s}-fixed', 'arm': a, 'seed': s, 'learning_rate': None, 'origin': 'fixed'}
                   for a in fixed for s in (8101, 8102, 8103)]
    identities += [{'key': f'{a}-{s}-cached', 'arm': a, 'seed': s, 'learning_rate': rates[a], 'origin': 'cached'}
                   for a in cached for s in (8101, 8102, 8103)]
    identities += [{'key': a, 'arm': a, 'seed': None, 'learning_rate': None, 'origin': 'reference'}
                   for a in ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')]
    parent_names = ['normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                    'causal_ridge_100.npz', 'causal_ridge_100.json', 'fits.json',
                    *[f'batches-{s}.npz' for s in (8101, 8102, 8103)]]
    parent_rates = {'last_two': 0, 'local_affine': 1, 'temporal_affine': 1, 'dense_bounded': 0,
                    'dense_unbounded': 1, 'gru32': 1, 'legacy_instant': 0, 'gru10': 1}
    parent_names += [f'{a}-{s}-lr{i}/final.npz' for a, i in parent_rates.items() for s in (8101, 8102, 8103)]
    dummy = {'sha256': 'a'*64, 'bytes': 1}
    plan = {'config': {'all_evaluation_data_exposed': True}, 'sources': dict.fromkeys(p.auditor.SOURCES, dummy),
            'inputs': {'parent/'+n: {'path': '/unused/'+n, **dummy} for n in parent_names}}
    attempts, rows = [], []
    for n in p.auditor.EXPOSED:
        for r in identities:
            common = {'recording': n, 'fit_key': r['key'], 'arm': r['arm'], 'seed': r['seed'], 'learning_rate': r['learning_rate']}
            attempts.append({**common, 'status': 'PASS', 'errors': [None, None], 'prediction_file': f"prediction-{n}-{r['key']}.npz"})
            for h in (64, 128):
                rows.append({**common, 'horizon': h, 'status': 'PASS', 'error': None,
                             'metrics': {'standardized_rmse': 1., 'standardized_sse': float(22*h*6), 'scalars': 22*h*6,
                                         'physical_rmse_deg': 2., 'per_joint_rmse_deg': [2.]*6, 'windows': 22, 'horizon': h}})
    selection = {'selected_rates': dict.fromkeys(learned, .001)}
    resources = [{**r, **p.auditor.storage(r['arm']), 'status': 'PASS', 'error': None,
                  'timing': {'seconds': [.004]*20, 'median_seconds': .004, 'p95_seconds': .004, 'scope': 'fabricated'}}
                 for r in identities if r['arm'] not in learned or r['learning_rate'] == .001]
    gate = {'status': 'OBSERVER_DEVELOPMENT_FAIL', 'passed': 3, 'total': 5,
            'conditions': [{'name': n, 'passed': i not in (1, 2)} for i, n in enumerate(p.auditor.CONDITIONS)],
            'details': {n: {'means': dict.fromkeys(p.FAMILIES, 1.)} for n in p.auditor.EXPOSED},
            'equal_file_means': dict.fromkeys(p.FAMILIES, 1.),
            'costs': {a: {'latency': .004, 'bytes': 4088} for a in p.FAMILIES}}
    value = {'results': {'version': 'robot-observer-study-v1', 'config': plan['config'], 'selection': selection, 'rows': rows, 'result': gate},
             'resources': resources, 'prediction_attempts': attempts, 'counts': {'metric_rows': 416}}
    names = {'registration.json', 'runtime.json', 'fits.json', 'checkpoint-barrier.json', 'parameter-checks.json',
             'prediction-attempts.json', 'results.json', 'resources.json', 'parent-parity.json'}
    names |= {'sources/'+n for n in plan['sources']} | set(plan['inputs'])
    for i, r in enumerate(identities[:48], 1):
        names.add(r['key']+'/final.npz')
        if i <= 18:
            names |= {r['key']+'/'+n for n in ('initial.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')}
            names.add(f'completed-fit-{i:02d}.json')
    names |= {f'completed-prediction-{i:03d}.json' for i in range(1, 209)}
    names |= {f'completed-timing-{i:02d}.json' for i in range(1, 44)}
    names |= {'dev-windows-'+n+'.npz' for n in p.auditor.EXPOSED}
    names |= {r['prediction_file'] for r in attempts}
    return plan, {name: dict(dummy) for name in names}, value


def test_complete_negative_roster_and_four_measurement_exclusions():
    plan, inventory, value = fixture()
    p.validate_scalars(value)
    excluded = p.study_roster(inventory, plan, value['prediction_attempts'])
    assert len(inventory) == 689 and len(excluded) == 4
    assert all(n.startswith('dev-windows-') for n in excluded)
    assert sum(n.endswith('/final.npz') and not n.startswith('parent/') for n in inventory) == 48
    assert sum(n.endswith('/initial.npz') for n in inventory) == 18
    assert sum(n.endswith('/optimizer.npz') for n in inventory) == 18
    assert sum(n.startswith('sources/') for n in inventory) == 45
    assert sum(n.startswith('parent/') for n in inventory) == 34
    assert len(value['results']['rows']) == 416 and len(value['resources']) == 43


@pytest.mark.parametrize('damage', ['checkpoint', 'optimizer', 'source', 'completion', 'target', 'parent', 'raw', 'alternate_target', 'traversal', 'absolute', 'attempt', 'bankname'])
def test_roster_rejects_omission_or_added_measurements(damage):
    plan, inventory, value = fixture()
    prefixes = {'checkpoint': 'last_two-8101-fixed/final.npz', 'optimizer': 'observer_learned-8101-lr0/optimizer.npz',
                'source': 'sources/'+p.auditor.SOURCES[0], 'completion': 'completed-prediction-208.json',
                'target': 'dev-windows-'+p.auditor.EXPOSED[0]+'.npz', 'parent': 'parent/normalizers.npz'}
    if damage in prefixes: inventory.pop(prefixes[damage])
    elif damage == 'attempt': value['prediction_attempts'].pop()
    elif damage == 'bankname': value['prediction_attempts'][0]['prediction_file'] = 'raw.mat'
    else: inventory[{'raw': 'raw.mat', 'alternate_target': 'targets.npz', 'traversal': '../raw.mat', 'absolute': '/raw.mat'}[damage]] = {'sha256': 'a'*64, 'bytes': 1}
    with pytest.raises(ValueError): p.study_roster(inventory, plan, value['prediction_attempts'])


def test_absent_forecast_preserves_failure_but_cannot_be_success():
    plan, inventory, value = fixture()
    attempt = value['prediction_attempts'][0]
    inventory.pop(attempt['prediction_file'])
    attempt.update(status='FAILED', prediction_file=None, errors=[{'type': 'FailedTrainingAttempt'}]*2)
    assert len(p.study_roster(inventory, plan, value['prediction_attempts'])) == 4
    attempt['status'] = 'PASS'
    with pytest.raises(ValueError): p.study_roster(inventory, plan, value['prediction_attempts'])


@pytest.mark.parametrize('damage', ['row', 'duplicate', 'rate', 'gate', 'promotion', 'nan', 'resource', 'hidden_failure', 'zero_cost', 'inf_cost', 'no_exposed'])
def test_scalar_coverage_and_outcome_integrity(damage):
    _, _, value = fixture()
    if damage == 'row': value['results']['rows'].pop()
    elif damage == 'duplicate': value['results']['rows'][-1] = value['results']['rows'][0]
    elif damage == 'rate': value['results']['rows'][0]['learning_rate'] = .9
    elif damage == 'gate': value['results']['result']['conditions'].pop()
    elif damage == 'promotion': value['results']['result']['status'] = 'OBSERVER_DEVELOPMENT_PASS'
    elif damage == 'nan': value['results']['rows'][0]['metrics']['standardized_rmse'] = float('nan')
    elif damage == 'resource': value['resources'].pop()
    elif damage == 'hidden_failure': value['results']['rows'][0]['status'] = 'FAILED'
    elif damage in ('zero_cost', 'inf_cost'): value['resources'][0]['timing']['median_seconds'] = 0. if damage == 'zero_cost' else float('inf')
    else: value['results']['config']['all_evaluation_data_exposed'] = False
    with pytest.raises(ValueError): p.validate_scalars(value)


def test_failed_and_unavailable_timing_remain_missing_not_zero(tmp_path):
    _, _, value = fixture()
    value['resources'][0].update(status='FAILED', timing=None, error={'type': 'NonfiniteTiming'})
    value['results']['rows'][0].update(status='FAILED', metrics=None, error={'type': 'KnownNumeric', 'message': 'preserved'})
    p.validate_scalars(value)
    p.write_tables(tmp_path, value)
    scores = list(csv.DictReader((tmp_path/'scores.csv').open()))
    costs = list(csv.DictReader((tmp_path/'costs.csv').open()))
    assert len(scores) == 416 and scores[0]['standardized_rmse'] == '' and scores[0]['error_message'] == 'preserved'
    assert len(costs) == 43 and costs[0]['median_seconds'] == ''
    assert sum(r['selected'] == 'False' for r in scores) == 72
    assert 'OBSERVER_DEVELOPMENT_FAIL' in (tmp_path/'table.md').read_text()
    assert 'All 416' in (tmp_path/'table.md').read_text()
    # Missing learned rate keeps all three explicit resource slots.
    value['results']['selection']['selected_rates']['observer_learned'] = None
    value['resources'] = [r for r in value['resources'] if r['arm'] != 'observer_learned']
    value['resources'] += [{**r, **p.auditor.storage(r['arm']), 'status': 'UNAVAILABLE', 'timing': None,
                           'error': {'type': 'UnavailableSelectedRecipe'}}
                          for r in p.auditor.resource_identities(value['results']['selection']) if r['origin'] == 'unavailable']
    p.validate_scalars(value)


def test_render_full_and_detail_keep_every_outlier_and_shared_axes(tmp_path, monkeypatch):
    _, _, value = fixture()
    # One selected and one unselected outlier in each file, plus an outlying mean.
    for name in p.auditor.EXPOSED:
        instances = [r for r in value['results']['rows'] if r['recording'] == name
                     and r['arm'] == 'local_affine' and r['horizon'] == 128]
        instances[0]['metrics']['standardized_rmse'] = 3.
        instances[1]['metrics']['standardized_rmse'] = 1e8
        value['results']['result']['details'][name]['means']['local_affine'] = 4.
    # Keep an explicit failed point visible in both views.
    value['results']['rows'][1].update(status='FAILED', metrics=None, error={'type': 'KnownNumeric'})
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
    p.render(tmp_path, value)
    assert value == untouched and len(figures) == 2
    (full_fig, full), (detail_fig, detail) = figures
    for axes in (full, detail):
        assert len({a.limits for a in axes.flat[:4]}) == 1 and axes.flat[0].limits[0] == 0
        assert [len(a.points) for a in axes.flat] == [51, 52, 52, 52, 43, 43]
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
    assert '3 of 5 criteria pass (all 5 required)' in full_fig.title
    assert 'See benchmark.png/SVG for the complete range' in detail_fig.texts[0]
    assert 'linear from 0 to 0.1' in full_fig.texts[0]


@pytest.mark.parametrize('name', ['../outside', '/outside', 'x/../outside', './x', '.', ''])
def test_safe_paths(name, tmp_path):
    with pytest.raises(ValueError): p.safe_path(tmp_path, name)


def test_tree_rejects_unlisted_files_and_symlinks(tmp_path):
    write(tmp_path/'kept', b'x'); write(tmp_path/'hidden/file', b'y')
    with pytest.raises(ValueError): p.regular_tree(tmp_path, {'kept'})
    (tmp_path/'hidden/file').unlink(); (tmp_path/'link').symlink_to(tmp_path/'kept')
    with pytest.raises(ValueError): p.regular_tree(tmp_path, {'kept', 'link'})


def scientific_qualifications(root, plan, statuses=('FAILED', 'PASS')):
    files = []
    commands = [['.venv/bin/ruff', 'check', *[n for n in p.auditor.QUALIFICATION_SOURCES if n.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_observer_initializer.py',
                 'tests/test_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py']]
    for i, status in enumerate(statuses, 1):
        folder = root/f'qualification-{i:02d}'
        sources = {n: write(folder/'sources'/n, f'{i}:{n}'.encode()) for n in p.auditor.SOURCES}
        launcher = write(folder/'sources'/p.LAUNCHER, f'launcher{i}'.encode())
        launch = {'created_utc': '2026-01-01T00:00:00+00:00', 'sources': sources, 'launcher': launcher, 'commands': [],
                  'thread_env': dict.fromkeys(p.auditor.THREADS, '1'), 'command_cap_seconds': 180, 'scope': 'fabricated'}
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
    plan, _, _ = fixture(); folders = scientific_qualifications(tmp_path, plan)
    files = p.qualification_files(tmp_path, plan)
    assert len(files) == 106  # 45 source files, launcher, seven original receipt/log/helper files, per attempt.
    assert files['engineering/qualification-01/receipt.json'] == folders[0]/'receipt.json'
    assert p.read(folders[0]/'receipt.json')['status'] == 'FAILED'
    assert p.read(folders[1]/'receipt.json')['status'] == 'PASS'


@pytest.mark.parametrize('damage', ['gap', 'snapshot', 'log', 'process', 'command', 'launcher', 'unlisted'])
def test_science_qualification_admission(damage, tmp_path):
    plan, _, _ = fixture(); folders = scientific_qualifications(tmp_path, plan)
    f = folders[0]
    if damage == 'gap': f.rename(tmp_path/'qualification-03')
    elif damage == 'snapshot': (f/'sources'/p.auditor.SOURCES[0]).write_bytes(b'changed')
    elif damage == 'log': (f/'command-01.log').write_bytes(b'changed')
    elif damage == 'process': write(f/'process-01.json', {})
    elif damage == 'launcher': (f/'sources'/p.LAUNCHER).write_bytes(b'changed')
    elif damage == 'unlisted': write(f/'surprise.npz', b'not a measurement')
    else:
        r = p.read(f/'receipt.json'); r['commands'][0]['command'] = ['echo', 'PASS']; write(f/'receipt.json', r)
        write(f/'process-01.json', r['commands'][0])
    with pytest.raises(ValueError): p.qualification_files(tmp_path, plan)


def publication_qualifications(root, statuses=('FAIL', 'PASS')):
    commands = [['.venv/bin/ruff', 'check', *p.DELIVERY_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', p.DELIVERY_SOURCES[1]]]
    folders = []
    for i, status in enumerate(statuses, 1):
        folder = root/p.PUBLICATION_ENGINEERING/f'qualification-{i:02d}'
        sources = {n: write(folder/'sources'/n, f'publication{i}:{n}'.encode()) for n in p.DELIVERY_SOURCES}
        for n in p.DELIVERY_SOURCES: write(root/n, (folder/'sources'/n).read_bytes())
        pre = {'sources': sources, 'commands': commands, 'thread_env': dict.fromkeys(p.auditor.THREADS, '1')}
        rows = []
        for j, cmd in enumerate(commands, 1):
            log = folder/f'command-{j:02d}.log'; d = write(log, f'{status}:{j}'.encode())
            rows.append({'command': cmd, 'returncode': 1 if status == 'FAIL' and j == 2 else 0, 'seconds': .1,
                         'log': str(log), 'sha256': d['sha256']})
        write(folder/'preflight.json', pre); write(folder/'receipt.json', {**pre, 'commands': rows, 'status': status, 'sources_unchanged': True})
        folders.append(folder)
    return folders


def test_publication_qualification_current_source_and_failed_history(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    folders = publication_qualifications(tmp_path)
    files = p.publication_qualification_files()
    assert len(files) == 12 and p.read(folders[0]/'receipt.json')['status'] == 'FAIL'
    (tmp_path/p.DELIVERY_SOURCES[0]).write_bytes(b'changed')
    with pytest.raises(ValueError): p.publication_qualification_files()


def test_publication_self_consistent_unrelated_commands_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    folder = publication_qualifications(tmp_path, ('PASS',))[0]
    pre, r = p.read(folder/'preflight.json'), p.read(folder/'receipt.json')
    pre['commands'][0] = ['echo', 'pass']; r['commands'][0]['command'] = ['echo', 'pass']
    write(folder/'preflight.json', pre); write(folder/'receipt.json', r)
    with pytest.raises(ValueError): p.publication_qualification_files()


def admitted_tree(root, monkeypatch):
    """Only inherited admission is stubbed; publication joins use real fake bytes."""
    plan, inventory, value = fixture()
    monkeypatch.setattr(p, 'ROOT', root)
    study, audit, eng = root/p.STUDY, root/p.AUDIT, root/p.ENGINEERING
    for name in inventory: write(study/name, b'opaque bytes, deliberately not NPZ')
    plan['qualification'] = {'path': str(eng/'qualification-01/receipt.json'), 'sha256': 'b'*64, 'bytes': 1}
    for name in ('parent_audit', 'parent_publication_manifest'):
        external = root/'old'/name; write(external, {'opaque': name}); plan[name] = pin(external)
    for name in p.DELIVERY_SOURCES: write(root/name, b'fabricated delivery source')
    write(root/p.LAUNCHER, b'fabricated launcher')
    source = root/'scripts/audit_robot_observer_study.py'; write(source, b'qualified auditor source')
    registration = root/'research/robot-observer-registration.json'; write(registration, plan)
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log'):
        write(eng/name, b'opaque admitted original process')
    write(study/'registration.json', plan)
    write(study/'results.json', value['results']); write(study/'resources.json', value['resources'])
    write(study/'prediction-attempts.json', value['prediction_attempts'])
    inventory = {name: p.descriptor(study/name) for name in inventory}
    write(study/'manifest.json', {'files': inventory}); write(study/'receipt.json', {'status': 'PASS'})
    inputs = {'registration': pin(registration), 'manifest': pin(study/'manifest.json')}
    value.update(status='PASS', agreement=True, study=str(study), registration_sha256=inputs['registration']['sha256'],
                 source_pins=plan['sources'], inputs=inputs, auditor=pin(source))
    write(audit, value); write(audit.parent/'manifest.json', {'files': {'audit.json': p.descriptor(audit)}})
    launch = {'command': ['.venv/bin/python', 'scripts/audit_robot_observer_study.py', '--study', p.STUDY,
                         '--run-receipt', p.ENGINEERING+'/run-process-01.json', '--output', p.AUDIT],
              'started_utc': '2026-01-01T00:00:00+00:00', 'thread_env': dict.fromkeys(p.auditor.THREADS, '1'), 'cap_seconds': 180}
    write(eng/'audit-launch-01.json', launch); log = write(eng/'audit-process-01.log', b'PASS fabricated')
    write(eng/'audit-process-01.json', {**launch, 'returncode': 0, 'external_timeout': False, 'elapsed_seconds': 2., 'log': log})
    closure = {'status': 'PASS', 'source_matches_registration': True, 'scope': 'postterminal exact original joins'}
    for key, path in {'auditor': source, 'audit_output': audit, 'audit_manifest': audit.parent/'manifest.json',
                      'process': eng/'audit-process-01.json', 'launch': eng/'audit-launch-01.json',
                      'log': eng/'audit-process-01.log', 'registration': registration}.items():
        closure[key] = pin(path)
    write(eng/'audit-closure-01.json', closure)
    write(eng/'registration-preflight-01.json', {'status': 'PASS', 'registration': inputs['registration'],
          'qualification': plan['qualification'], 'sources': 45, 'inputs': 45, 'model_calls': 0, 'numeric_decodes': 0, 'optimizer_updates': 0})
    monkeypatch.setattr(p.auditor, 'authenticate', lambda *args: (copy.deepcopy(plan), copy.deepcopy(inputs)))
    monkeypatch.setattr(p, 'qualification_files', lambda *args: {})
    monkeypatch.setattr(p, 'publication_qualification_files', dict)
    return study, audit, eng, plan, value


def test_metadata_admission_joins_exact_successful_audit_and_saved_scalars(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    auth = p.authenticate(study, audit, eng)
    assert len(auth['excluded']) == 4 and len(auth['inventory']) == 689
    assert auth['audit']['results']['result']['status'] == 'OBSERVER_DEVELOPMENT_FAIL'


@pytest.mark.parametrize('damage', ['failed', 'timeout', 'overcap', 'nan_time', 'command', 'log', 'closure', 'inputs', 'disagreement', 'manifest', 'source', 'result'])
def test_original_audit_admission_failures(damage, tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    process = eng/'audit-process-01.json'
    if damage in ('failed', 'timeout', 'overcap', 'nan_time', 'command'):
        record = p.read(process)
        if damage == 'failed': record['returncode'] = 1
        elif damage == 'timeout': record['external_timeout'] = True
        elif damage == 'overcap': record['elapsed_seconds'] = 181.
        elif damage == 'nan_time': record['elapsed_seconds'] = float('nan')
        else: record['command'][1] = 'unrelated.py'
        write(process, record)
    elif damage == 'log': write(eng/'audit-process-01.log', b'changed')
    elif damage == 'source': write(tmp_path/'scripts/audit_robot_observer_study.py', b'changed')
    elif damage == 'closure': write(eng/'audit-closure-01.json', {'status': 'FAIL'})
    elif damage == 'manifest': write(audit.parent/'manifest.json', {'files': {}})
    elif damage == 'result':
        result = p.read(study/'results.json'); result['result']['passed'] = 5; write(study/'results.json', result)
    else:
        value = p.read(audit)
        if damage == 'inputs': value['inputs']['manifest']['sha256'] = 'c'*64
        else: value['agreement'] = False
        write(audit, value)
        write(audit.parent/'manifest.json', {'files': {'audit.json': p.descriptor(audit)}})
        closure = p.read(eng/'audit-closure-01.json'); closure['audit_output'] = pin(audit); closure['audit_manifest'] = pin(audit.parent/'manifest.json')
        write(eng/'audit-closure-01.json', closure)
    with pytest.raises(ValueError): p.authenticate(study, audit, eng)


def test_publication_full_opaque_roundtrip_and_no_measurement_copies(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(p, 'render', lambda output, value: (write(output/'benchmark.png', b'fakePNG'), write(output/'benchmark.svg', b'fakeSVG')))
    output = tmp_path/p.OUTPUT
    receipt = p.publish(study, audit, eng, output)
    manifest = p.read(output/'manifest.json')
    assert receipt['scientific_status'] == 'OBSERVER_DEVELOPMENT_FAIL' and receipt['passed'] == 3
    assert receipt['model_calls'] == receipt['array_decodes'] == receipt['rescoring_calls'] == receipt['timing_calls'] == 0
    assert len(manifest['excluded_measurement_payloads']) == 4
    assert not list((output/'study').glob('dev-windows-*.npz'))
    assert sum(n.startswith('study/') and n.endswith('/final.npz') for n in manifest['files']) == 72  # 48 child plus 24 inherited.
    assert 'scores.csv' in manifest['files'] and 'costs.csv' in manifest['files']
    for name, item in manifest['copy_sources'].items():
        assert (output/name).read_bytes() == Path(item['path']).read_bytes()
        assert p.descriptor(output/name) == {k: item[k] for k in ('bytes', 'sha256')}
    with pytest.raises(ValueError): p.publish(study, audit, eng, output)


def test_copy_source_mutation_prevents_success_receipt(tmp_path, monkeypatch):
    study, audit, eng, _, _ = admitted_tree(tmp_path, monkeypatch)
    def changed(output, value):
        (study/'last_two-8101-fixed/final.npz').write_bytes(b'changed after admission')
    monkeypatch.setattr(p, 'render', changed)
    with pytest.raises(ValueError): p.publish(study, audit, eng, tmp_path/p.OUTPUT)
    assert not (tmp_path/p.OUTPUT/'receipt.json').exists()
