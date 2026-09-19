"""Synthetic publication orchestration with no real subprocess or network calls.

Package receipts/assets are constructed stand-ins, not actual scientific files
or archives. Pure byte/manifest checks use the real hash-pinned package helpers.
Git/GitHub/curl are mocked; these tests do not establish live publication success.
"""

import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).with_name('publish_release.py')
    spec = importlib.util.spec_from_file_location('synthetic_geometry_memory_publisher', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, passed=False):
    v = module()
    original_base = v.BASE
    base = tmp_path / 'output/reacher-geometry-memory-v1'
    base.mkdir(parents=True)
    for name in ('publish_release.py', 'package.py'):
        shutil.copyfile(original_base / name, base / name)
    monkeypatch.setattr(v, 'ROOT', tmp_path)
    monkeypatch.setattr(v, 'BASE', base)
    monkeypatch.setattr(v, '__file__', str(base / 'publish_release.py'))
    # No accidental external action is allowed even if mock dispatch misses it.
    monkeypatch.setattr(v.subprocess, 'run', lambda *a, **k: pytest.fail('Unmocked subprocess'))
    args = SimpleNamespace(publish_authorized=True, package=base / 'publication-v1',
        expected_package_receipt_sha256='', expected_audit_receipt_sha256='a' * 64,
        target_commit='c' * 40, release_notes=base / 'notes.md', expected_release_notes_sha256='',
        title='Synthetic completed comparison', out=tmp_path / 'evidence/scored-publication')
    args.package.mkdir()
    args.release_notes.write_text('Synthetic failed or passed gate is retained.\n')
    args.expected_release_notes_sha256 = v.sha(args.release_notes)
    sources = {}
    for index in range(90):
        path = tmp_path / f'sources/{index:03d}.py'
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'VALUE = {index}\n')
        sources[path.relative_to(tmp_path).as_posix()] = v.sha(path)
    helper = v.load_package()
    archives, assets = [], []
    for index, name in enumerate(v.ARCHIVES):
        path = args.package / name
        path.write_bytes(b'constructed asset bytes ' + str(index).encode())
        members = []
        for source in ((base / 'publish_release.py', base / 'package.py') if index == 0 else (tmp_path / 'sources/000.py',)):
            members.append({'path': source.relative_to(tmp_path).as_posix(),
                'bytes': source.stat().st_size, 'sha256': v.sha(source)})
        archive = {'name': name, 'scope': f'fixture-{index}', 'bytes': path.stat().st_size,
            'sha256': v.sha(path), 'member_count': len(members), 'members': members,
            'all_members_reopened_and_verified': True, 'split_for_release': False,
            'concatenation_verified': False, 'ordered_release_assets': [name]}
        archives.append(archive)
        assets.append({'name': name, 'bytes': path.stat().st_size, 'sha256': v.sha(path),
            'bundle': name, 'scope': archive['scope']})
    dependency = {'scope': 'explicit synthetic prior release dependency'}
    manifest = {'study': v.STUDY, 'archives': archives, 'previous_release_dependency': dependency}
    manifest_sha = save(args.package / 'manifest.json', manifest)
    package = {'status': 'verified', 'study': v.STUDY, 'plan_sha256': v.PLAN_SHA,
        'readiness_sha256': helper.EXPECTED_READINESS, 'audit_receipt_sha256': args.expected_audit_receipt_sha256,
        'package_source_sha256': v.PACKAGE_SHA, 'new_model_calls': 0, 'new_native_calls': 0,
        'new_optimizer_steps': 0, 'uploaded': False, 'release_created': False,
        'reporting_receipt_sha256': '1' * 64, 'terminal_verification_sha256': '2' * 64,
        'previous_release_verification_sha256': '3' * 64, 'execution_completed_sha256': '4' * 64,
        'raw_coverage': copy.deepcopy(v.RAW_COVERAGE), 'source_sha256': sources,
        'manifest_sha256': manifest_sha, 'archives': [{k: value for k, value in row.items() if k != 'members'} for row in archives],
        'previous_release_dependency': dependency, 'member_count': 3, 'release_assets': assets,
        'scientific_status': {'checks_total': 25, 'checks_passed': 25 if passed else 17,
            'continuation_passed': passed, 'packaging_changes_gate': False}}
    save(args.package / 'packaging-started.json', {'requested_audit_receipt_sha256': args.expected_audit_receipt_sha256,
        'package_source_sha256': v.PACKAGE_SHA})
    (args.package / 'README.md').write_text('Synthetic package scope. No real archive was made.\n')
    seal(v, args, package)
    return v, args, package


def seal(v, args, package):
    args.expected_package_receipt_sha256 = save(args.package / 'receipt.json', package)
    names = [row['name'] for row in package['release_assets']] + list(v.SIDECARS[:-1])
    (args.package / 'SHA256SUMS').write_text(''.join(f'{v.sha(args.package / name)}  {name}\n' for name in names))


def remote(v, args, monkeypatch, *, bad_digest=False, upload_failure=False, tag_changed=False, bad_sidecar=False):
    events, state = [], {'published': False}
    names = [*[row['name'] for row in json.loads((args.package / 'receipt.json').read_text())['release_assets']], *v.SIDECARS]
    assets = [{'name': name, 'id': index + 100, 'state': 'uploaded',
        'size': (args.package / name).stat().st_size, 'digest': 'sha256:' + v.sha(args.package / name),
        'browser_download_url': f'https://github.com/{v.REPO}/releases/download/{v.TAG}/{name}'}
        for index, name in enumerate(names)]
    if bad_digest:
        assets[0]['digest'] = 'sha256:' + '0' * 64

    def release():
        return {'id': 9, 'tag_name': v.TAG, 'target_commitish': args.target_commit,
            'draft': not state['published'], 'prerelease': False, 'name': args.title,
            'body': args.release_notes.read_text(), 'published_at': 'synthetic-time' if state['published'] else None,
            'html_url': f'https://github.com/{v.REPO}/releases/tag/{v.TAG}'}

    def api(*items):
        events.append(('api', items))
        if items == ('user',):
            return {'login': 'kw2828'}
        if items[0] == f'repos/{v.REPO}/commits/{args.target_commit}':
            return {'sha': args.target_commit}
        if items[0] == f'repos/{v.REPO}/git/ref/heads/main':
            return {'object': {'sha': args.target_commit}}
        if '/assets?' in items[0]:
            return [copy.deepcopy(assets)]
        if items[:2] == ('--method', 'POST'):
            assert not state['published']
            return release()
        if items[:2] == ('--method', 'PATCH'):
            state['published'] = True
            return release()
        if items == (f'repos/{v.REPO}/releases/9',):
            return release()
        pytest.fail(f'Unexpected mocked API call {items}')

    tag_calls = []

    def tag():
        tag_calls.append(True)
        return 'd' * 40 if tag_changed and len(tag_calls) > 1 else args.target_commit

    def run(argv, **kwargs):
        events.append(('run', tuple(argv)))
        if argv[:2] == ['git', 'show']:
            _, relative = argv[2].split(':', 1)
            return SimpleNamespace(stdout=(v.ROOT / relative).read_bytes())
        if argv[:3] == ['gh', 'release', 'upload']:
            assert '--clobber' not in argv
            if upload_failure:
                raise subprocess.CalledProcessError(1, argv)
            return SimpleNamespace(returncode=0)
        if argv[0] == 'curl':
            destination = Path(argv[argv.index('--output') + 1])
            source = args.package / argv[-1].rsplit('/', 1)[-1]
            destination.write_bytes(b'bad' if bad_sidecar else source.read_bytes())
            return SimpleNamespace(returncode=0)
        pytest.fail(f'Unexpected mocked subprocess {argv}')

    monkeypatch.setattr(v, 'api', api)
    monkeypatch.setattr(v, 'tag_target', tag)
    monkeypatch.setattr(v, 'release_absent', lambda: events.append(('absence', True)))
    monkeypatch.setattr(v.subprocess, 'run', run)
    return events, state


@pytest.mark.parametrize('passed', [False, True])
def test_mocked_publication_retains_gate_and_verifies_all_assets(tmp_path, monkeypatch, passed):
    v, args, package = fixture(tmp_path, monkeypatch, passed=passed)
    events, state = remote(v, args, monkeypatch)
    result = v.publish(args)
    assert state['published'] and result['scientific_status'] == package['scientific_status']
    assert result['status'] == 'published_and_verified' and len(result['assets']) == 7
    assert len(result['public_sidecar_downloads']) == 3 and result['new_model_calls'] == 0
    assert json.loads((args.out / 'release-verification.json').read_text()) == result
    upload = [i for i, event in enumerate(events) if event[0] == 'run' and event[1][:3] == ('gh', 'release', 'upload')]
    patches = [i for i, event in enumerate(events) if event[0] == 'api' and event[1][:2] == ('--method', 'PATCH')]
    assert len(upload) == len(patches) == 1 and upload[0] < patches[0]


def test_multipart_release_uploads_ordered_segments_and_keeps_complete_local_archive(tmp_path, monkeypatch):
    v, args, package = fixture(tmp_path, monkeypatch)
    archive = package['archives'][0]
    data = (args.package / archive['name']).read_bytes()
    parts = []
    for index, payload in enumerate((data[:10], data[10:])):
        path = args.package / f"{archive['name']}.part{index:03d}"
        path.write_bytes(payload)
        parts.append({'name': path.name, 'bytes': len(payload), 'sha256': v.sha(path),
            'bundle': archive['name'], 'scope': archive['scope']})
    archive.update(split_for_release=True, concatenation_verified=True,
        ordered_release_assets=[row['name'] for row in parts])
    package['release_assets'] = parts + package['release_assets'][1:]
    manifest = json.loads((args.package / 'manifest.json').read_text())
    manifest['archives'][0].update(archive)
    package['manifest_sha256'] = save(args.package / 'manifest.json', manifest)
    seal(v, args, package)
    events, _ = remote(v, args, monkeypatch)
    result = v.publish(args)
    upload = next(event[1] for event in events if event[0] == 'run' and event[1][:3] == ('gh', 'release', 'upload'))
    assert str(args.package / archive['name']) not in upload
    assert [row['name'] for row in result['assets']][:2] == archive['ordered_release_assets']
    assert (args.package / archive['name']).read_bytes() == data


@pytest.mark.parametrize('kind', ['pushed_main', 'committed_source'])
def test_exact_pushed_commit_and_committed_helper_bytes_required(tmp_path, monkeypatch, kind):
    v, args, _ = fixture(tmp_path, monkeypatch)
    events, state = remote(v, args, monkeypatch)
    original_api, original_run = v.api, v.subprocess.run

    def changed_api(*items):
        if items == (f'repos/{v.REPO}/git/ref/heads/main',):
            return {'object': {'sha': 'd' * 40}}
        return original_api(*items)

    def changed_run(argv, **kwargs):
        if argv[:2] == ['git', 'show']:
            return SimpleNamespace(stdout=b'other committed source')
        return original_run(argv, **kwargs)

    if kind == 'pushed_main':
        monkeypatch.setattr(v, 'api', changed_api)
    else:
        monkeypatch.setattr(v.subprocess, 'run', changed_run)
    with pytest.raises(ValueError, match='(pushed main|publication sources)'):
        v.publish(args)
    assert not state['published']
    assert not any(event[0] == 'api' and event[1][:2] == ('--method', 'POST') for event in events)


@pytest.mark.parametrize('case', ['digest', 'upload', 'tag', 'sidecar'])
def test_failure_is_preserved_without_retry_or_gate_change(tmp_path, monkeypatch, case):
    v, args, package = fixture(tmp_path, monkeypatch)
    events, state = remote(v, args, monkeypatch, bad_digest=case == 'digest', upload_failure=case == 'upload',
        tag_changed=case == 'tag', bad_sidecar=case == 'sidecar')
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        v.publish(args)
    assert not (args.out / 'release-verification.json').exists()
    assert json.loads((args.out / 'publication-failed.json').read_text())['status'] == 'failed'
    assert state['published'] is (case == 'sidecar')
    assert package['scientific_status']['continuation_passed'] is False
    assert sum(event[0] == 'run' and event[1][:3] == ('gh', 'release', 'upload') for event in events) == 1


@pytest.mark.parametrize('mutation', ['coverage', 'audit_hash', 'gate', 'missing_publisher', 'asset', 'extra_file'])
def test_invalid_package_rejected_before_remote_reads(tmp_path, monkeypatch, mutation):
    v, args, package = fixture(tmp_path, monkeypatch)
    if mutation == 'coverage':
        package['raw_coverage']['inherited_fits'] = 11
    elif mutation == 'audit_hash':
        args.expected_audit_receipt_sha256 = 'b' * 64
    elif mutation == 'gate':
        package['scientific_status']['continuation_passed'] = True
    elif mutation == 'missing_publisher':
        manifest = json.loads((args.package / 'manifest.json').read_text())
        manifest['archives'][0]['members'][0]['path'] = 'other/publisher.py'
        package['manifest_sha256'] = save(args.package / 'manifest.json', manifest)
    elif mutation == 'asset':
        (args.package / v.ARCHIVES[0]).write_bytes(b'changed asset')
    elif mutation == 'extra_file':
        (args.package / 'packaging-failed.json').write_text('{}')
    seal(v, args, package)
    monkeypatch.setattr(v, 'api', lambda *a: pytest.fail('Remote read before local authentication'))
    with pytest.raises(ValueError):
        v.publish(args)
    assert (args.out / 'publication-failed.json').exists()


def test_authorization_and_existing_output_guards(tmp_path, monkeypatch):
    v, args, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(v, 'authenticate_package', lambda *a: pytest.fail('Authentication before guard'))
    args.publish_authorized = False
    with pytest.raises(ValueError, match='authorization'):
        v.publish(args)
    assert not args.out.exists()
    args.publish_authorized = True
    args.out.mkdir(parents=True)
    (args.out / 'original').write_text('unchanged')
    with pytest.raises(ValueError, match='Exclusive'):
        v.publish(args)
    assert (args.out / 'original').read_text() == 'unchanged'


@pytest.mark.parametrize('stdout', ['{"status":"500","message":"error"}', 'bad response'])
def test_release_absence_requires_explicit_404(tmp_path, monkeypatch, stdout):
    v = module()
    monkeypatch.setattr(v.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout=stdout))
    with pytest.raises(ValueError):
        v.release_absent()


def test_hash_pinned_packager_ignores_timestamp_valid_stale_bytecode(tmp_path, monkeypatch):
    v = module()
    path = tmp_path / 'package.py'
    prefix = f'STUDY={v.STUDY!r}\nEXPECTED_PLAN={v.PLAN_SHA!r}\n'
    path.write_text(prefix + "MARKER='stale'\n")
    before = path.stat()
    py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    path.write_text(prefix + "MARKER='fresh'\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    control_spec = importlib.util.spec_from_file_location('ordinary_cached_packager', path)
    control = importlib.util.module_from_spec(control_spec)
    control_spec.loader.exec_module(control)
    assert control.MARKER == 'stale'
    monkeypatch.setattr(v, 'BASE', tmp_path)
    with pytest.raises(ValueError, match='source hash'):
        v.load_package()
    monkeypatch.setattr(v, 'PACKAGE_SHA', v.sha(path))
    assert v.load_package().MARKER == 'fresh'
