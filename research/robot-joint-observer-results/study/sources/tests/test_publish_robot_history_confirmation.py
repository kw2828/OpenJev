"""Synthetic scalar and opaque-byte publication fixtures; no model or decoder."""
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import publish_robot_history_confirmation as publication


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return publication.descriptor(path)


def pin(path):
    return {'path': str(path.resolve()), **publication.descriptor(path)}


def fixture():
    a = publication.auditor
    source_pins = {name: {'sha256': hashlib.sha256(name.encode()).hexdigest(), 'bytes': len(name)} for name in a.SOURCES}
    plan = {'config': a.config(), 'sources': source_pins}
    attempts, rows = [], []
    for name in a.CONFIRM:
        identities = [(a.model_key(arm, seed), arm, seed, a.FIXED_RATES[arm]) for arm in a.ARMS for seed in a.SEEDS]
        identities += [(arm, arm, None, None) for arm in a.REFS]
        for key, arm, seed, rate in identities:
            common = {'recording': name, 'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate}
            filename = f'prediction-{name}-{key}.npz'
            attempts.append({**common, 'status': 'PASS', 'errors': [None, None],
                             'prediction_file': filename, 'prediction_pin': {'sha256': 'a'*64, 'bytes': 1}})
            for horizon in (64, 128):
                rows.append({**common, 'horizon': horizon, 'status': 'PASS', 'error': None,
                             'metrics': {'standardized_rmse': 1., 'standardized_sse': 22*horizon*6.,
                                         'scalars': 22*horizon*6, 'physical_rmse_deg': 2.,
                                         'per_joint_rmse_deg': [2.]*6, 'windows': 22, 'horizon': horizon}})
    resources = [{'arm': arm, 'seed': seed, 'learning_rate': a.FIXED_RATES.get(arm),
                  'status': 'PASS', 'error': None, 'timing': {'median_seconds': .004}}
                 for arm in publication.FAMILIES for seed in (a.SEEDS if arm in a.ARMS else (None,))]
    gate = {'status': 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION', 'passed': 3, 'total': 5,
            'conditions': [{'name': n, 'passed': i not in (1, 2)} for i, n in enumerate(a.CONDITIONS)],
            'details': {name: {'means': dict.fromkeys(publication.FAMILIES, 1.)} for name in a.CONFIRM},
            'equal_file_means': dict.fromkeys(publication.FAMILIES, 1.),
            'costs': {arm: {'latency': .004, 'bytes': 4088} for arm in publication.FAMILIES}}
    value = {'results': {'config': a.config(), 'fixed_rates': dict(a.FIXED_RATES), 'rows': rows, 'result': gate},
             'resources': resources, 'prediction_attempts': attempts}
    names = {'registration.json', 'runtime.json', 'models.json', 'checkpoint-barrier.json', 'parameter-checks.json',
             'prediction-attempts.json', 'results.json', 'resources.json', *a.COMMON_PAYLOADS}
    names |= {'sources/' + name for name in a.SOURCES}
    names |= {a.model_key(arm, seed) + '/final.npz' for arm in a.ARMS for seed in a.SEEDS}
    names |= {f'completed-prediction-{i:02d}.json' for i in range(1, 57)}
    names |= {f'completed-timing-{i:02d}.json' for i in range(1, 29)}
    names |= {f'{prefix}-{name}.npz' for prefix in ('confirm-data', 'confirm-windows') for name in a.CONFIRM}
    names |= {r['prediction_file'] for r in attempts}
    return plan, {name: {'sha256': 'a'*64, 'bytes': 1} for name in names}, value


def test_complete_negative_result_keeps_all_cases_and_exactly_four_measurement_exclusions():
    plan, inventory, value = fixture()
    publication.validate_scalars(value)
    excluded = publication.study_roster(inventory, plan, value['prediction_attempts'])
    assert len(inventory) == 218 and len(excluded) == 4
    assert all(name.startswith(('confirm-data-', 'confirm-windows-')) for name in excluded)
    assert len(value['results']['rows']) == 112 and len(value['resources']) == 28
    assert sum(name.endswith('/final.npz') for name in inventory) == 24
    assert sum(name.startswith('sources/') for name in inventory) == 36
    assert value['results']['result']['status'] == 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION'


@pytest.mark.parametrize('damage', ['missing_checkpoint', 'missing_completion', 'missing_target', 'unknown_measurement', 'raw', 'traversal', 'absolute', 'changed_pin'])
def test_opaque_study_roster_rejects_omissions_and_unsafe_payloads(damage):
    plan, inventory, value = fixture()
    remove = {'missing_checkpoint': 'last_two-8101-lr0/final.npz', 'missing_completion': 'completed-timing-28.json',
              'missing_target': 'confirm-data-' + publication.auditor.CONFIRM[0] + '.npz'}
    if damage in remove:
        inventory.pop(remove[damage])
    elif damage == 'changed_pin':
        inventory[value['prediction_attempts'][0]['prediction_file']]['bytes'] = 2
    else:
        name = {'unknown_measurement': 'confirmation-targets.npz', 'raw': 'raw/recording.mat',
                'traversal': '../outside', 'absolute': '/outside'}[damage]
        inventory[name] = {'sha256': 'a'*64, 'bytes': 1}
    with pytest.raises(ValueError): publication.study_roster(inventory, plan, value['prediction_attempts'])


def test_guard_failure_can_omit_only_its_bank_without_omitting_attempt_or_scores():
    plan, inventory, value = fixture()
    attempt = value['prediction_attempts'][0]
    inventory.pop(attempt['prediction_file'])
    attempt.update(status='FAILED', prediction_file=None, prediction_pin=None,
                   errors=[{'type': 'NonfiniteEvaluation', 'message': 'nonfinite ridge prediction'}]*2)
    assert len(publication.study_roster(inventory, plan, value['prediction_attempts'])) == 4
    attempt['status'] = 'PASS'
    with pytest.raises(ValueError): publication.study_roster(inventory, plan, value['prediction_attempts'])


@pytest.mark.parametrize('damage', ['row', 'duplicate', 'rate', 'condition', 'promotion', 'nan', 'resource', 'hidden_failure'])
def test_saved_scalar_schema_prevents_omitted_cases_or_outcome_promotion(damage):
    _, _, value = fixture()
    if damage == 'row': value['results']['rows'].pop()
    elif damage == 'duplicate': value['results']['rows'][-1] = value['results']['rows'][0]
    elif damage == 'rate': value['results']['rows'][0]['learning_rate'] = .5
    elif damage == 'condition': value['results']['result']['conditions'].pop()
    elif damage == 'promotion': value['results']['result']['status'] = 'CONFIRMED_HISTORY_INITIALIZATION'
    elif damage == 'nan': value['results']['rows'][0]['metrics']['standardized_rmse'] = float('nan')
    elif damage == 'resource': value['resources'].pop()
    else: value['results']['rows'][0]['status'] = 'FAILED'
    with pytest.raises(ValueError): publication.validate_scalars(value)


def test_csv_keeps_failed_rows_and_h64_without_recomputing_scores(tmp_path):
    _, _, value = fixture()
    value['results']['rows'][0].update(status='FAILED', metrics=None, error={'type': 'KnownNumeric', 'message': 'kept'})
    publication.write_tables(tmp_path, value)
    with (tmp_path / 'scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 112 and rows[0]['status'] == 'FAILED' and rows[0]['standardized_rmse'] == ''
    assert rows[0]['error_message'] == 'kept'
    assert rows[1]['horizon'] == '128' and rows[1]['physical_rmse_deg'] == '2.0'
    assert 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION' in (tmp_path / 'table.md').read_text()
    assert '3/5' in (tmp_path / 'table.md').read_text()


def test_renderer_has_common_accuracy_scale_log_costs_and_all_instance_dots(tmp_path, monkeypatch):
    _, _, value = fixture()
    class Axis:
        def __init__(self): self.limits = None; self.scale = None; self.points = []; self.label = ''
        def scatter(self, x, y, **kwargs): self.points.append((x, y))
        def plot(self, *args, **kwargs): pass
        def text(self, *args, **kwargs): pass
        def get_xaxis_transform(self): return None
        def set_title(self, *args): pass
        def set_ylabel(self, label): self.label = label
        def set_ylim(self, *args, **kwargs): self.limits = args
        def set_yscale(self, value): self.scale = value
        def set_xticks(self, *args, **kwargs): pass
        def grid(self, *args, **kwargs): pass
        def set_axisbelow(self, *args): pass
    class Axes(list):
        def __getitem__(self, key):
            return super().__getitem__(key[0])[key[1]] if isinstance(key, tuple) else super().__getitem__(key)
        @property
        def flat(self): return [axis for row in self for axis in row]
    class Figure:
        def suptitle(self, text, **kwargs): assert text == 'Fixed-checkpoint confirmation: 3 of 5 criteria pass (all 5 required)'
        def text(self, *args, **kwargs): pass
        def tight_layout(self, **kwargs): pass
        def savefig(self, path, **kwargs): write(path, b'figure stub')
    axes = Axes([[Axis(), Axis()], [Axis(), Axis()]])
    module = types.ModuleType('matplotlib'); module.use = lambda backend: None
    pyplot = types.ModuleType('matplotlib.pyplot')
    pyplot.subplots = lambda *args, **kwargs: (Figure(), axes)
    pyplot.close = lambda figure: None
    monkeypatch.setitem(sys.modules, 'matplotlib', module)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    publication.render(tmp_path, value)
    assert axes[0][0].limits == axes[0][1].limits == (0, 1.08)
    assert all(axis.scale == 'log' and 'log scale' in axis.label for axis in axes[1])
    assert len(axes[0][0].points) == len(axes[0][1].points) == 28
    assert len(axes[1][0].points) == 28
    assert {y for _, y in axes[1][0].points} == {4.}
    assert (tmp_path/'benchmark.png').is_file() and (tmp_path/'benchmark.svg').is_file()


def engineering_fixture(root, plan):
    a = publication.auditor
    engineering = root / publication.ENGINEERING
    commands = [['.venv/bin/ruff', 'check', *a.QUALIFICATION_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                 'tests/test_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py']]
    for number in (1, 2):
        folder = engineering / f'qualification-{number:02d}'
        sources = {}
        for name in a.SOURCES:
            blob = (root / name).read_bytes()
            if number == 1 and name == 'tests/test_robot_history_confirmation.py': blob += b' old fixture'
            sources[name] = write(folder / 'sources' / name, blob)
        launcher = write(folder / 'sources' / publication.LAUNCHER, (root / publication.LAUNCHER).read_bytes())
        pre = {'sources': sources, 'launcher': launcher, 'commands': commands}
        write(folder / 'preflight.json', pre)
        command_rows = []
        for i, command in enumerate(commands, 1):
            log = folder / f'command-{i:02d}.log'; log_pin = write(log, f'original {number}/{i}'.encode())
            command_rows.append({'command': command, 'returncode': int(number == 1 and i == 2), 'log': str(log),
                                 'sha256': log_pin['sha256'], 'seconds': .1})
        write(folder / 'receipt.json', {'sources': sources, 'launcher': launcher, 'sources_unchanged': True,
              'thread_env': dict.fromkeys(a.THREADS, '1'), 'commands': command_rows, 'status': 'FAIL' if number == 1 else 'PASS'})
    plan['qualification'] = pin(engineering / 'qualification-02/receipt.json')
    return engineering


def publication_qualification_fixture(root):
    folder = root/publication.PUBLICATION_ENGINEERING/'qualification-01'
    sources = {name: write(folder/'sources'/name, (root/name).read_bytes()) for name in publication.DELIVERY_SOURCES}
    commands = [['.venv/bin/ruff', 'check', *publication.DELIVERY_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', publication.DELIVERY_SOURCES[1]]]
    write(folder/'preflight.json', {'sources': sources, 'commands': commands})
    rows = []
    for i, command in enumerate(commands, 1):
        log = folder/f'command-{i:02d}.log'; digest = write(log, b'original publication qualification')
        rows.append({'command': command, 'returncode': 0, 'log': str(log), 'sha256': digest['sha256'], 'seconds': .1})
    write(folder/'receipt.json', {'sources': sources, 'sources_unchanged': True, 'status': 'PASS',
          'thread_env': dict.fromkeys(publication.auditor.THREADS, '1'), 'commands': rows})


def opaque_fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve(); monkeypatch.setattr(publication, 'ROOT', root)
    plan, inventory, value = fixture()
    for name in (*publication.auditor.SOURCES, *publication.DELIVERY_SOURCES, publication.LAUNCHER):
        write(root / name, ('source ' + name).encode())
    plan['sources'] = {name: publication.descriptor(root / name) for name in publication.auditor.SOURCES}
    plan['launcher'] = publication.descriptor(root / publication.LAUNCHER)
    engineering = engineering_fixture(root, plan)
    publication_qualification_fixture(root)
    plan['parent_closure'] = {name: {'path': str(root / 'parent' / (name+'.json')), **write(root / 'parent' / (name+'.json'), {'original': name})}
                              for name in ('manifest', 'receipt', 'process', 'audit', 'audit_process')}
    plan.update(archive={'path': '/external/archive.rar', 'sha256': 'e'*64, 'bytes': 10},
                raw_recordings={name: {'path': '/external/'+name, 'sha256': 'f'*64, 'bytes': 20} for name in publication.auditor.CONFIRM},
                extraction={'path': '/external/extraction.json', 'sha256': 'd'*64, 'bytes': 30})
    registration = root / 'research/robot-history-confirmation-registration.json'; write(registration, plan)
    study, audit_path = root / publication.STUDY, root / publication.AUDIT
    for name in inventory:
        write(study / name, b'opaque never an array')
    for name in plan['sources']:
        write(study / 'sources' / name, (root/name).read_bytes())
    for row in value['prediction_attempts']:
        row['prediction_pin'] = publication.descriptor(study / row['prediction_file'])
    write(study / 'results.json', value['results']); write(study / 'resources.json', value['resources'])
    write(study / 'prediction-attempts.json', value['prediction_attempts'])
    write(study / 'registration.json', plan)
    inventory = {name: publication.descriptor(study / name) for name in inventory}
    write(study / 'manifest.json', {'files': inventory}); write(study / 'receipt.json', {'status': 'PASS'})
    for name in ('run-launch-01.json', 'run-process-01.json'): write(engineering/name, {'original': name})
    write(engineering / 'run-process-01.log', b'original run log')
    inputs = {'registration': pin(registration), 'manifest': pin(study / 'manifest.json')}
    value.update(status='PASS', agreement=True, study=str(study), registration_sha256=inputs['registration']['sha256'],
                 inputs=inputs, source_pins=plan['sources'], auditor=pin(root / 'scripts/audit_robot_history_confirmation.py'))
    write(audit_path, value); write(audit_path.parent / 'manifest.json', {'files': {'audit.json': publication.descriptor(audit_path)}})
    command = ['.venv/bin/python', 'scripts/audit_robot_history_confirmation.py', '--study', publication.STUDY,
               '--run-receipt', publication.ENGINEERING + '/run-process-01.json', '--output', publication.AUDIT]
    launch = {'command': command, 'started_utc': '2026-09-25T00:00:00+00:00',
              'thread_env': dict.fromkeys(publication.auditor.THREADS, '1'), 'cap_seconds': 180}
    write(engineering/'audit-launch-01.json', launch)
    log = write(engineering/'audit-process-01.log', b'original audit log')
    write(engineering/'audit-process-01.json', {**launch, 'returncode': 0, 'elapsed_seconds': .1, 'external_timeout': False, 'log': log})
    closure = {'scope': 'Postterminal binding, not a replacement original receipt', 'status': 'PASS', 'source_matches_registration': True}
    for key, path in {'auditor': root / 'scripts/audit_robot_history_confirmation.py', 'audit_output': audit_path,
                      'audit_manifest': audit_path.parent / 'manifest.json', 'process': engineering/'audit-process-01.json',
                      'launch': engineering/'audit-launch-01.json', 'log': engineering/'audit-process-01.log', 'registration': registration}.items():
        closure[key] = pin(path)
    write(engineering/'audit-closure-01.json', closure)
    def metadata_only(folder, process):
        assert folder == study and process == engineering / 'run-process-01.json'
        return plan, inputs
    monkeypatch.setattr(publication.auditor, 'authenticate', metadata_only)
    monkeypatch.setattr(publication, '__file__', str(root / publication.DELIVERY_SOURCES[0]))
    return plan, study, audit_path, engineering


def test_opaque_admission_joins_separate_postterminal_closure_and_all_qualifications(tmp_path, monkeypatch):
    plan, study, audit_path, engineering = opaque_fixture(tmp_path, monkeypatch)
    auth = publication.authenticate(study, audit_path, engineering)
    assert auth['plan'] == plan and len(auth['excluded']) == 4
    assert len(auth['qualification_files']) == 88  # Two science attempts, plus six publication qualification files.
    q1 = engineering/'qualification-01/sources/tests/test_robot_history_confirmation.py'
    q2 = engineering/'qualification-02/sources/tests/test_robot_history_confirmation.py'
    assert q1.read_bytes() != q2.read_bytes()


@pytest.mark.parametrize('damage', ['failed', 'timeout', 'over_cap', 'command', 'log', 'closure_output', 'closure_source', 'manifest'])
def test_original_audit_admission_fails_before_parent_or_scalar_admission(tmp_path, monkeypatch, damage):
    _, study, audit_path, engineering = opaque_fixture(tmp_path, monkeypatch)
    def forbidden(*args): raise AssertionError('must reject before parent admission')
    monkeypatch.setattr(publication.auditor, 'authenticate', forbidden)
    process_path = engineering/'audit-process-01.json'; process = publication.read(process_path)
    if damage == 'failed': process['returncode'] = 1
    elif damage == 'timeout': process['external_timeout'] = True
    elif damage == 'over_cap': process['elapsed_seconds'] = 180.01
    elif damage == 'command': process['command'][1] = 'unrelated.py'
    elif damage == 'log': write(engineering/'audit-process-01.log', b'changed')
    elif damage in ('closure_output', 'closure_source'):
        path = engineering/'audit-closure-01.json'; closure = publication.read(path)
        closure['audit_output' if damage == 'closure_output' else 'auditor']['sha256'] = '0'*64; write(path, closure)
    else: write(audit_path.parent/'manifest.json', {'files': {}})
    if damage in ('failed', 'timeout', 'over_cap', 'command'): write(process_path, process)
    with pytest.raises(ValueError): publication.authenticate(study, audit_path, engineering)


@pytest.mark.parametrize('damage', ['dropped_original', 'wrong_command', 'old_source_mutation', 'extra_file', 'symlink'])
def test_qualification_attempts_cannot_be_replaced_or_silently_trimmed(tmp_path, monkeypatch, damage):
    plan, _, _, engineering = opaque_fixture(tmp_path, monkeypatch)
    folder = engineering/'qualification-01'
    if damage == 'dropped_original': (folder/'command-02.log').unlink()
    elif damage == 'wrong_command':
        for name in ('preflight.json', 'receipt.json'):
            record = publication.read(folder/name)
            if name == 'preflight.json': record['commands'][0] = ['echo', 'PASS']
            else: record['commands'][0]['command'] = ['echo', 'PASS']
            write(folder/name, record)
    elif damage == 'old_source_mutation': write(folder/'sources/tests/test_robot_history_confirmation.py', b'new fabricated source')
    elif damage == 'extra_file': write(folder/'unlisted.txt', b'hidden')
    else: (folder/'alias').symlink_to(folder/'receipt.json')
    with pytest.raises(ValueError): publication.qualification_files(engineering, plan)


def test_publication_qualification_binds_exact_current_source_and_commands(tmp_path, monkeypatch):
    opaque_fixture(tmp_path, monkeypatch)
    assert len(publication.publication_qualification_files()) == 6
    source = tmp_path/publication.DELIVERY_SOURCES[0]
    write(source, b'mutated publisher')
    with pytest.raises(ValueError): publication.publication_qualification_files()


def test_complete_opaque_copy_excludes_all_measurements_and_preserves_negative_verdict(tmp_path, monkeypatch):
    _, study, audit_path, engineering = opaque_fixture(tmp_path, monkeypatch)
    def fake_render(output, value):
        assert value['results']['result']['passed'] == 3
        write(output/'benchmark.png', b'fabricated PNG'); write(output/'benchmark.svg', b'fabricated SVG')
    monkeypatch.setattr(publication, 'render', fake_render)
    output = tmp_path/publication.OUTPUT
    receipt = publication.publish(study, audit_path, engineering, output)
    assert receipt['status'] == 'PASS' and receipt['scientific_status'] == 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION'
    assert receipt['model_calls'] == receipt['array_decodes'] == receipt['rescoring_calls'] == receipt['timing_calls'] == 0
    manifest = publication.read(output/'manifest.json')
    assert len(manifest['excluded_measurement_payloads']) == 4
    assert not any((output/'study'/name).exists() for name in manifest['excluded_measurement_payloads'])
    assert not any(p.suffix in ('.mat', '.rar') for p in output.rglob('*'))
    assert sum(name.startswith('study/prediction-') and name.endswith('.npz') for name in manifest['files']) == 56
    assert sum(name.endswith('/final.npz') for name in manifest['files']) == 24
    for name, item in manifest['copy_sources'].items():
        assert (output/name).read_bytes() == Path(item['path']).read_bytes()
    for name, item in manifest['files'].items():
        assert publication.descriptor(output/name) == item
    assert publication.descriptor(output/'manifest.json') == receipt['manifest']
    assert '3/5' in (output/'README.md').read_text()
    with pytest.raises(ValueError): publication.publish(study, audit_path, engineering, output)


def test_copy_time_input_mutation_leaves_failed_attempt_without_success_receipt(tmp_path, monkeypatch):
    _, study, audit_path, engineering = opaque_fixture(tmp_path, monkeypatch)
    def mutate(output, value):
        write(study/'results.json', {'changed': True})
        write(output/'benchmark.png', b'placeholder'); write(output/'benchmark.svg', b'placeholder')
    monkeypatch.setattr(publication, 'render', mutate)
    output = tmp_path/publication.OUTPUT
    with pytest.raises(ValueError): publication.publish(study, audit_path, engineering, output)
    assert output.exists() and not (output/'receipt.json').exists()


def test_safe_path_rejects_symlinks_and_traversal(tmp_path):
    write(tmp_path/'original', b'a')
    (tmp_path/'alias').symlink_to(tmp_path/'original')
    for name in ('../escape', '/absolute', 'alias', './original', '.'):
        with pytest.raises(ValueError): publication.safe_path(tmp_path, name)


def test_failed_process_does_not_create_publication_directory(tmp_path, monkeypatch):
    def fail(*args): raise ValueError('original audit not closed')
    monkeypatch.setattr(publication, 'authenticate', fail)
    output = tmp_path/'never-created'
    with pytest.raises(ValueError): publication.publish(tmp_path, tmp_path/'audit', tmp_path, output)
    assert not output.exists()
