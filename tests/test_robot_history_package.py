"""Opaque publication fixtures only: no numerical decoder, model or campaign."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import package_robot_history_initialization as package

PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
CACHED = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*PRIMARY, *CACHED)
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS = (8101, 8102, 8103)
DEV = ('fabricated-A', 'fabricated-B')
FIT_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')


def pin(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return pin(path)


def roster_fixture():
    keys = [f'{arm}-{seed}-lr{ri}' for arm in ARMS for seed in SEEDS for ri in (0, 1)]
    sources = {f'src/fabricated-source-{i:02d}.py': {'sha256': '1' * 64, 'bytes': 1} for i in range(29)}
    names = {'registration.json', 'runtime.json', 'normalizers.npz', 'linear.npz',
             'causal_ridge_1.npz', 'causal_ridge_1.json', 'causal_ridge_100.npz', 'causal_ridge_100.json',
             'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json', 'permutation.json',
             'parent-structured-fits.json', 'parent-structured-results.json',
             'parent-transition-fits.json', 'parent-transition-results.json'}
    names |= {f'batches-{seed}.npz' for seed in SEEDS}
    names |= {f'completed-fit-{index:02d}.json' for index in range(1, 49)}
    names |= {f'{key}/{name}' for key in keys for name in FIT_FILES}
    names |= {'sources/' + name for name in sources}
    names |= {f'dev-windows-{name}.npz' for name in DEV}
    names |= {f'prediction-{name}-{key}.npz' for name in DEV for key in keys}
    names |= {f'prediction-{name}-{arm}.npz' for name in DEV for arm in REFERENCES}
    names |= {f'permuted-prediction-{name}-{arm}-{seed}-lr0.npz'
              for name in DEV for arm in PRIMARY for seed in SEEDS}
    plan = {'config': {'seeds': list(SEEDS), 'learning_rates': [.001, .003], 'comparison_arms': list(ARMS),
                       'partitions': {'dev': list(DEV)}}, 'sources': sources}
    return plan, {name: {'sha256': hashlib.sha256(name.encode()).hexdigest(), 'bytes': 1}
                  for name in sorted(names)}


def test_safe_roster_retains_all_attempts_and_exactly_two_target_exclusions():
    plan, original = roster_fixture()
    excluded = package.study_roster(original, plan)
    targets = {f'dev-windows-{name}.npz' for name in DEV}
    assert excluded == {name: original[name] for name in targets}
    assert len(original) == 461
    assert sum(name.endswith('/final.npz') for name in original) == 48
    assert sum(name.startswith('completed-fit-') for name in original) == 48
    assert sum(name.startswith('sources/') for name in original) == 29
    assert sum(name.startswith('prediction-') for name in original) == 104
    assert sum(name.startswith('permuted-prediction-') for name in original) == 18


def test_failed_forecasts_may_be_absent_without_losing_any_fit_evidence():
    plan, original = roster_fixture()
    for name in list(original):
        if name.startswith(('prediction-fabricated-A-temporal_affine-', 'permuted-prediction-')):
            del original[name]
    assert len(package.study_roster(original, plan)) == 2
    assert sum(name.endswith('/final.npz') for name in original) == 48
    assert sum(name.startswith('prediction-') for name in original) == 98


@pytest.mark.parametrize('damage', ['checkpoint', 'completion', 'source', 'target', 'reference',
                                   'raw', 'target_alias', 'wrong_family', 'wrong_seed', 'diagnostic_control',
                                   'traversal', 'absolute'])
def test_safe_roster_rejects_omitted_evidence_and_unregistered_paths(damage):
    plan, original = roster_fixture()
    remove = {'checkpoint': 'last_two-8101-lr0/optimizer.npz', 'completion': 'completed-fit-48.json',
              'source': 'sources/' + next(iter(plan['sources'])), 'target': 'dev-windows-fabricated-A.npz',
              'reference': 'prediction-fabricated-A-persistence.npz'}
    if damage in remove:
        original.pop(remove[damage])
    else:
        extra = {'raw': 'dev-data-measured.npz', 'target_alias': 'targets.npz',
                 'wrong_family': 'prediction-fabricated-A-householder-8101-lr0.npz',
                 'wrong_seed': 'prediction-fabricated-A-last_two-8104-lr0.npz',
                 'diagnostic_control': 'permuted-prediction-fabricated-A-gru32-8101-lr0.npz',
                 'traversal': '../outside.json', 'absolute': '/outside.json'}[damage]
        original[extra] = {'sha256': '0' * 64, 'bytes': 0}
    with pytest.raises(ValueError):
        package.study_roster(original, plan)


def test_closed_run_and_audit_admission_precede_reads_and_output_creation(tmp_path, monkeypatch):
    calls = []
    def denied(*args):
        calls.append('original closure admission')
        raise ValueError('independent audit has not closed')
    monkeypatch.setattr(package.plotter, 'authenticate', denied)
    monkeypatch.setattr(package.plotter, 'read_json', lambda *args: pytest.fail('read before admission'))
    output = tmp_path / 'public'
    with pytest.raises(ValueError, match='not closed'):
        package.package(tmp_path / 'study', tmp_path / 'audit.json', tmp_path / 'plot',
                        tmp_path / 'engineering', output)
    assert calls == ['original closure admission'] and not output.exists()


def plot_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    study, plot, engineering = [root / name for name in ('study', 'plot', 'engineering')]
    source = root / 'scripts/plot_robot_history_initialization.py'
    write(source, b'# fabricated plot source')
    monkeypatch.setattr(package.plotter, 'ROOT', root)
    monkeypatch.setattr(package.plotter, '__file__', str(source))
    outputs = ('benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'permutation.csv', 'table.md', 'plotted-values.json')
    for name in outputs:
        write(plot / name, ('opaque plot artifact ' + name).encode())
    auth = {'study': str(study), 'audit_path': str(root / 'audit/audit.json'),
            'inputs': {str(study / 'manifest.json'): {'sha256': '2' * 64, 'bytes': 100}}}
    receipt = {'version': package.plotter.VERSION, 'study': str(study),
               'registration_sha256': package.plotter.PLAN_SHA, 'inputs': copy.deepcopy(auth['inputs']),
               'script': pin(source), 'outputs': {name: pin(plot / name) for name in outputs}}
    write(plot / 'receipt.json', receipt)
    process = {'returncode': 0, 'source_unchanged': True, 'plot_receipt': pin(plot / 'receipt.json'), 'script': pin(source),
               'log': write(engineering / 'plot-process-01.log', b'original fabricated plot log'),
               'command': ['.venv/bin/python', str(source), '--study', str(study), '--output', str(plot),
                           '--audit', auth['audit_path'], '--engineering', str(engineering)]}
    write(engineering / 'plot-process-01.json', process)
    return plot, engineering, auth, receipt, process, source


def test_plot_admission_joins_all_six_outputs_audit_inputs_and_original_process(tmp_path, monkeypatch):
    plot, engineering, auth, receipt, _, _ = plot_fixture(tmp_path, monkeypatch)
    assert package.plot_admission(plot, engineering, auth) == receipt
    assert set(receipt['outputs']) == {'benchmark.png', 'benchmark.pdf', 'all-candidates.csv',
                                      'permutation.csv', 'table.md', 'plotted-values.json'}


@pytest.mark.parametrize('damage', ['audit_inputs', 'study', 'registration', 'version', 'output_pin',
                                   'extra_output', 'directory', 'source', 'log', 'process_receipt',
                                   'process_failure', 'argv_script', 'argv_study', 'argv_audit', 'argv_output', 'symlink'])
def test_plot_admission_rejects_provenance_substitution(tmp_path, monkeypatch, damage):
    plot, engineering, auth, receipt, process, source = plot_fixture(tmp_path, monkeypatch)
    if damage == 'audit_inputs': receipt['inputs'] = {}
    elif damage == 'study': receipt['study'] = str(tmp_path / 'other')
    elif damage == 'registration': receipt['registration_sha256'] = '0' * 64
    elif damage == 'version': receipt['version'] = 'different-renderer'
    elif damage == 'output_pin': receipt['outputs']['table.md']['bytes'] += 1
    elif damage == 'extra_output': write(plot / 'unlisted.txt', b'not admitted')
    elif damage == 'directory': write(plot / 'hidden/unlisted.txt', b'not admitted')
    elif damage == 'source': source.write_bytes(b'changed after rendering')
    elif damage == 'log': (engineering / 'plot-process-01.log').write_bytes(b'changed log')
    elif damage == 'process_receipt': process['plot_receipt']['sha256'] = '0' * 64
    elif damage == 'process_failure': process['returncode'] = 1
    elif damage.startswith('argv_'):
        position = {'argv_script': 1, 'argv_study': 3, 'argv_output': 5, 'argv_audit': 7}[damage]
        process['command'][position] = str(tmp_path / 'other')
    else:
        original = plot / 'table.md'; outside = tmp_path / 'alias.md'
        outside.write_bytes(original.read_bytes()); original.unlink(); original.symlink_to(outside)
    write(plot / 'receipt.json', receipt)
    if damage in ('audit_inputs', 'study', 'registration', 'version', 'output_pin'):
        process['plot_receipt'] = pin(plot / 'receipt.json')
    write(engineering / 'plot-process-01.json', process)
    with pytest.raises(ValueError):
        package.plot_admission(plot, engineering, auth)


def attempt_fixture(root, folder, sources, codes=(0, 0), revision='current'):
    sources = {name: write(root / name, (revision + ':' + name).encode()) for name in sources}
    for name in sources:
        write(folder / 'sources' / name, (root / name).read_bytes())
    commands = [['.venv/bin/ruff', 'check', *sources],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', *[n for n in sources if n.startswith('tests/')]]]
    threads = dict.fromkeys(('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
    pre = {'sources': sources, 'thread_env': threads, 'commands': commands}
    rows = []
    for index, code in enumerate(codes):
        log = folder / f'command-{index+1}.log'
        rows.append({'command': commands[index], 'returncode': code, 'seconds': .1,
                     'log': str(log), 'log_pin': write(log, f'original output status {code}'.encode())})
    receipt = {'status': 'PASS' if codes == (0, 0) else 'FAILED', 'sources': sources,
               'sources_unchanged': True, 'thread_env': threads, 'commands': rows}
    write(folder / 'preflight.json', pre); write(folder / 'receipt.json', receipt)
    return pre, receipt


@pytest.mark.parametrize('codes,files', [((1,), 5), ((0, 1), 6), ((0, 0), 6)])
def test_qualification_preserves_only_executed_logs_and_exact_source_snapshots(tmp_path, monkeypatch, codes, files):
    root, folder = tmp_path / 'repo', tmp_path / 'attempt'
    monkeypatch.setattr(package.plotter, 'ROOT', root)
    attempt_fixture(root, folder, package.AUDIT_SOURCES, codes)
    paths = package.qualification_attempt(folder, package.AUDIT_SOURCES)
    assert len(paths) == files
    assert (folder / 'command-2.log') in paths if len(codes) == 2 else not (folder / 'command-2.log').exists()
    if codes == (0, 0):
        assert package.qualification_attempt(folder, package.AUDIT_SOURCES, require_current=True) == paths
    else:
        with pytest.raises(ValueError, match='final qualification'):
            package.qualification_attempt(folder, package.AUDIT_SOURCES, require_current=True)


@pytest.mark.parametrize('damage', ['source', 'snapshot', 'log', 'extra_log', 'source_roster', 'thread',
                                   'wrong_command', 'self_consistent_unrelated_command', 'continuation_after_failure',
                                   'status', 'duration', 'log_path', 'symlink'])
def test_qualification_rejects_mutation_or_invented_history(tmp_path, monkeypatch, damage):
    root, folder = tmp_path / 'repo', tmp_path / 'attempt'
    monkeypatch.setattr(package.plotter, 'ROOT', root)
    pre, receipt = attempt_fixture(root, folder, package.AUDIT_SOURCES)
    source_name = package.AUDIT_SOURCES[0]
    if damage == 'source': (root / source_name).write_bytes(b'changed current source')
    elif damage == 'snapshot': (folder / 'sources' / source_name).write_bytes(b'changed snapshot')
    elif damage == 'log': (folder / 'command-1.log').write_bytes(b'changed output')
    elif damage == 'extra_log': write(folder / 'command-3.log', b'invented retry')
    elif damage == 'source_roster': pre['sources'] = {}
    elif damage == 'thread': receipt['thread_env'] = {}
    elif damage == 'wrong_command': receipt['commands'][0]['command'] = ['other']
    elif damage == 'self_consistent_unrelated_command':
        pre['commands'][0] = receipt['commands'][0]['command'] = ['.venv/bin/python', '-c', 'pass']
    elif damage == 'continuation_after_failure': receipt['commands'][0]['returncode'] = 1
    elif damage == 'status': receipt['status'] = 'FAILED'
    elif damage == 'duration': receipt['commands'][0]['seconds'] = 0.
    elif damage == 'log_path': receipt['commands'][0]['log'] = str(tmp_path / 'unrelated.log')
    else:
        source = folder / 'sources' / source_name; source.unlink(); source.symlink_to(root / source_name)
    write(folder / 'preflight.json', pre); write(folder / 'receipt.json', receipt)
    with pytest.raises(ValueError):
        package.qualification_attempt(folder, package.AUDIT_SOURCES, require_current=True)


def qualification_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    monkeypatch.setattr(package.plotter, 'ROOT', root)
    scientific = root / 'research/robot-history-initialization-qualification-results'
    files = {f'original-{index:02d}.bin': write(scientific / f'original-{index:02d}.bin', f'original evidence {index}'.encode())
             for index in range(72)}
    write(scientific / 'manifest.json', {'files': files})
    monkeypatch.setattr(package, 'SCIENTIFIC_QUALIFICATION_SHA', pin(scientific / 'manifest.json')['sha256'])
    audit = root / 'output/robot-history-initialization-audit-engineering-v1/qualification-01'
    attempt_fixture(root, audit, package.AUDIT_SOURCES)
    public = root / 'output/robot-history-publication-engineering-v1'
    attempt_fixture(root, public / 'qualification-01', package.PUBLICATION_SOURCES, (1,), 'failed-source')
    attempt_fixture(root, public / 'qualification-02', package.PUBLICATION_SOURCES, (0, 0), 'corrected-source')
    return root, scientific, audit, public


def test_exact_published_scientific_qualification_pin_is_not_ambient_latest():
    assert package.SCIENTIFIC_QUALIFICATION_SHA == '29d1b0a7b9683e2cef2f3c35b9d825cbadb3fe450c39c0fabcd8d3dae3f06130'


def test_qualification_bundle_retains_failed_attempt_and_different_original_source_bytes(tmp_path, monkeypatch):
    root, _, _, _ = qualification_fixture(tmp_path, monkeypatch)
    files = package.qualification_files()
    assert len(files) == 73 + 6 + 7 + 8
    assert sum(name.startswith('scientific-qualification/') for name in files) == 73
    first = files['publication-qualification/qualification-01/receipt.json']
    assert json.loads(first.read_text())['status'] == 'FAILED'
    assert 'publication-qualification/qualification-01/command-2.log' not in files
    source = package.PUBLICATION_SOURCES[0]
    old = files['publication-qualification/qualification-01/sources/' + source]
    new = files['publication-qualification/qualification-02/sources/' + source]
    assert old.read_bytes() != new.read_bytes() == (root / source).read_bytes()


@pytest.mark.parametrize('damage', ['bundle_hash', 'bundle_payload', 'omitted_attempt', 'missing_audit', 'last_failed'])
def test_qualification_bundle_cannot_drop_original_failure_or_promote_failed_final(tmp_path, monkeypatch, damage):
    _, scientific, audit, public = qualification_fixture(tmp_path, monkeypatch)
    if damage == 'bundle_hash': (scientific / 'manifest.json').write_bytes(b'{}')
    elif damage == 'bundle_payload': (scientific / 'original-00.bin').write_bytes(b'changed retained evidence')
    elif damage == 'omitted_attempt': (public / 'qualification-02').rename(public / 'qualification-03')
    elif damage == 'missing_audit': (audit / 'command-1.log').unlink()
    else:
        path = public / 'qualification-02/receipt.json'; value = json.loads(path.read_text())
        value['status'] = 'FAILED'; value['commands'][-1]['returncode'] = 1; write(path, value)
    with pytest.raises(ValueError): package.qualification_files()


def copy_fixture(tmp_path, monkeypatch):
    root, _, _, _ = qualification_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(package, '__file__', str(root / 'scripts/package_robot_history_initialization.py'))
    study, plot, engineering, audit_folder, output = [tmp_path / name for name in ('study', 'plot', 'engineering', 'audit', 'public')]
    plan, original = roster_fixture()
    for name in original:
        original[name] = write(study / name, ('not an array: ' + name).encode())
    write(study / 'manifest.json', {'files': original})
    write(study / 'receipt.json', {'status': 'PASS'})
    audit_path = audit_folder / 'audit.json'
    write(audit_path, {'status': 'PASS', 'agreement': True})
    write(audit_folder / 'manifest.json', {'files': {'audit.json': pin(audit_path)}})
    for name in (*package.PLOT_FILES, 'receipt.json'):
        write(plot / name, ('opaque plot bytes ' + name).encode())
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log',
                 'audit-process-01.json', 'audit-process-01.log', 'plot-process-01.json', 'plot-process-01.log'):
        write(engineering / name, ('opaque original closure ' + name).encode())
    # Qualification attempts already own the six publication/auditor sources.
    write(root / 'scripts/launch_robot_history_initialization.py', b'# original launcher')
    plan['parent_closure'] = {}
    for group in ('structured', 'transition'):
        write(root / f'research/robot-{group}-registration.json', {'original': group})
        plan['parent_closure'][group] = {}
        for name in ('receipt', 'manifest', 'process', 'audit', 'audit_process'):
            path = tmp_path / 'ancestors' / group / (name + '.json')
            plan['parent_closure'][group][name] = {'path': str(path), **write(path, {'original': group + name})}
    plan['data'] = {'external-saved-recording.npz': {'path': str(tmp_path / 'must-not-be-read.npz'),
                                                   'sha256': '4' * 64, 'bytes': 100}}
    conditions = [{'name': name, 'passed': passed} for name, passed in zip(package.plotter.CONDITIONS,
                  (True, False, True, True, False), strict=True)]
    auth = {'study': str(study), 'audit_path': str(audit_path), 'plan': plan,
            'audit': {'results': {'result': {'status': 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION',
                                            'passed': 3, 'total': 5, 'conditions': conditions}}}}
    monkeypatch.setattr(package.plotter, 'authenticate', lambda *args: copy.deepcopy(auth))
    monkeypatch.setattr(package, 'plot_admission', lambda *args: {'opaque_fixture': True})
    return study, audit_path, plot, engineering, output, plan, original


def test_complete_opaque_roundtrip_preserves_failure_label_and_every_original_byte(tmp_path, monkeypatch):
    study, audit_path, plot, engineering, output, plan, original = copy_fixture(tmp_path, monkeypatch)
    before = {path: pin(path) for path in tmp_path.rglob('*') if path.is_file()}
    receipt = package.package(study, audit_path, plot, engineering, output)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert receipt['manifest'] == pin(output / 'manifest.json')
    assert receipt['files'] == len(manifest['files']) + 1
    assert receipt['bytes'] == sum(item['bytes'] for item in manifest['files'].values())
    assert set(manifest['files']) == {str(path.relative_to(output)) for path in output.rglob('*')
                                    if path.is_file() and path != output / 'manifest.json'}
    for relative, expected in manifest['files'].items(): assert pin(output / relative) == expected
    for relative, descriptor in manifest['copy_sources'].items():
        expected = {key: descriptor[key] for key in ('sha256', 'bytes')}
        assert pin(output / relative) == pin(descriptor['path']) == expected
    assert all(pin(path) == value for path, value in before.items())
    targets = {f'dev-windows-{name}.npz' for name in DEV}
    assert receipt['excluded_targets'] == 2
    assert manifest['excluded_target_payloads'] == {name: original[name] for name in targets}
    assert all(not (output / 'study' / name).exists() for name in targets)
    assert manifest['external_measurement_pins'] == plan['data']
    assert sum(name.startswith('study/prediction-') for name in manifest['files']) == 104
    assert sum(name.startswith('study/permuted-prediction-') for name in manifest['files']) == 18
    assert sum(name.startswith('study/sources/') for name in manifest['files']) == 29
    assert sum(name.startswith('study/') and name.endswith('/final.npz') for name in manifest['files']) == 48
    assert sum(name.startswith('scientific-qualification/') for name in manifest['files']) == 73
    failed = output / 'publication-qualification/qualification-01/receipt.json'
    assert json.loads(failed.read_text())['status'] == 'FAILED'
    assert 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION' in (output / 'README.md').read_text()
    assert '3/5 conditions' in (output / 'README.md').read_text()
    with pytest.raises(FileExistsError): package.package(study, audit_path, plot, engineering, output)


def test_changed_source_during_copy_cannot_receive_complete_manifest(tmp_path, monkeypatch):
    study, audit_path, plot, engineering, output, _, _ = copy_fixture(tmp_path, monkeypatch)
    original_copy = package.shutil.copyfileobj
    changed = []
    def mutate_after_copy(reader, writer):
        original_copy(reader, writer)
        if not changed and reader.name.endswith('/initial.npz'):
            Path(reader.name).write_bytes(b'changed original evidence during copy')
            changed.append(reader.name)
    monkeypatch.setattr(package.shutil, 'copyfileobj', mutate_after_copy)
    with pytest.raises(ValueError, match='byte-exact'):
        package.package(study, audit_path, plot, engineering, output)
    assert len(changed) == 1 and output.exists() and not (output / 'manifest.json').exists()


@pytest.mark.parametrize('inside', ['study', 'plot', 'engineering', 'audit', 'ancestor'])
def test_package_output_cannot_overlap_immutable_inputs(tmp_path, monkeypatch, inside):
    study, audit_path, plot, engineering, _, _, _ = copy_fixture(tmp_path, monkeypatch)
    destination = {'study': study / 'publication', 'plot': plot / 'publication',
                   'engineering': engineering / 'publication', 'audit': audit_path.parent / 'publication',
                   'ancestor': tmp_path}[inside]
    with pytest.raises(ValueError, match='immutable evidence'):
        package.package(study, audit_path, plot, engineering, destination)
