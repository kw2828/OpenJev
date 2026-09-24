"""Package closed rounded-learning evidence with an opaque-byte roundtrip.

Only authenticated inputs and explicitly named presentation files are included.
No array, model, optimizer, or world generator is loaded by this helper.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from render_finite_rounded_learning import (
    REGISTRATION_SHA256,
    ROOT,
    STUDY_NAME,
    authenticate,
    descriptor,
    inventory,
    read,
    regular,
    require,
)

VERSION = 'finite-rounded-learning-package-v1'
ASSETS = (STUDY_NAME + '.tar.gz', 'manifest.json', 'package-receipt.json', 'SHA256SUMS')


def safe_relative(name):
    require(type(name) is str and name and '\\' not in name, 'portable member name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
            'safe relative member')
    return path.as_posix()


def committed_blob(path, commit):
    name = regular(path).relative_to(ROOT).as_posix()
    entries = subprocess.check_output(['git', 'ls-tree', '-z', commit, '--', name],
                                      cwd=ROOT, timeout=30).split(b'\0')
    entries = [entry for entry in entries if entry]
    require(len(entries) == 1, 'one committed publication file: ' + name)
    header, saved_name = entries[0].split(b'\t', 1)
    mode, kind, blob = header.decode().split()
    require(saved_name.decode() == name and kind == 'blob' and mode in ('100644', '100755'),
            'regular committed blob')
    require(subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=ROOT,
                                    timeout=30) == path.read_bytes(),
            'publication commit matches bytes: ' + name)
    return {'git_blob': blob, **descriptor(path)}


def package(study, report, output, *, registration_sha256, commit, publication):
    study, report, output = Path(study), Path(report), Path(output)
    require(registration_sha256 == REGISTRATION_SHA256, 'owner-supplied exact registration')
    require(output.is_absolute() and output.resolve() == output and not output.exists(),
            'exclusive absolute package directory')
    require(report == ROOT / 'research/finite-rounded-learning-results',
            'exact completed report folder')
    require(re.fullmatch('[0-9a-f]{40}', commit)
            and subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'],
                                        cwd=ROOT, text=True, timeout=30).strip() == commit,
            'existing full publication commit')
    auth = authenticate(study)
    plan = auth['plan']
    receipt = read(report / 'receipt.json')
    outputs = {'summary.json', 'report.md', 'benchmark.png'}
    require(set(receipt['files']) == outputs and inventory(report) == {
        **receipt['files'], 'receipt.json': descriptor(report / 'receipt.json')},
        'complete renderer output hashes')
    require(receipt['registration'] == {'path': str(auth['plan_path']),
                                       **descriptor(auth['plan_path'])},
            'renderer joins exact registration')
    require(receipt['renderer'] == {'path': str(ROOT / 'scripts/render_finite_rounded_learning.py'),
             **descriptor(ROOT / 'scripts/render_finite_rounded_learning.py')},
            'renderer source matches receipt')
    require(receipt['version'] == 'finite-rounded-learning-report-v1'
            and receipt['inputs'] == auth['inputs'] and receipt['sources'] == plan['sources']
            and receipt['kernel_qualification'] == plan['kernel_qualification']
            and receipt['result_reads_after_original_closure'] is True
            and receipt['counts'] == dict.fromkeys(('array_decodes', 'checkpoint_decodes',
                'model_calls', 'generator_calls', 'optimizer_calls', 'native_calls', 'teacher_calls'), 0)
            and receipt['model_selection'] is False, 'metadata-only publication joins closed evidence')

    members, origins = {}, {}

    def add(name, path, expected=None):
        name, path = safe_relative(name), regular(path)
        require(path.is_relative_to(ROOT) and name != 'MANIFEST.json', 'repository-bound file')
        actual = descriptor(path)
        require(expected is None or expected == actual, 'authenticated input: ' + name)
        if name in members:
            require(origins[name] == path and members[name]['sha256'] == actual['sha256'],
                    'unambiguous duplicate member')
            return
        members[name], origins[name] = {**actual, 'original_path': str(path)}, path

    for path, pin in auth['inputs'].items():
        path = Path(path)
        add('evidence/' + path.relative_to(ROOT).as_posix(), path, pin)
    for phase in auth['phases'].values():
        launch = phase['launch_path']
        log = launch.with_name(launch.name.removesuffix('.launch.json') + '.log')
        add('evidence/' + log.relative_to(ROOT).as_posix(), log)
    for name in sorted(outputs | {'receipt.json'}):
        add('publication/report/' + name, report / name)
    # The two registrations each retain the exact sources as they existed at freeze.
    qualification_plan = regular(Path(auth['phases']['qualify']['receipt']['plan']))
    require(sorted(study.glob('engineering-registration-[0-9][0-9].json')) == [qualification_plan]
            and qualification_plan.name == 'engineering-registration-01.json',
            'complete first-pass engineering registration roster')
    require(len(plan['sources']) == 63, 'exact registered source roster')
    snapshots = {}
    for mode, registration_path in (('engineering', qualification_plan), ('study', auth['plan_path'])):
        registration = read(registration_path)
        require(registration['mode'] == mode and registration['sources'] == plan['sources'],
                'both source snapshots belong to these closed registrations')
        snapshot = study / ('source-snapshot-' + mode + '-01')
        manifest_path = snapshot / 'manifest.json'
        require(read(manifest_path) == {
            'registration': {'path': str(registration_path), **descriptor(registration_path)},
            'sources': plan['sources']}, 'source snapshot joins exact original registration')
        pins = {**plan['sources'], 'manifest.json': descriptor(manifest_path)}
        require(inventory(snapshot) == pins, 'complete source snapshot including manifest')
        for name, pin in sorted(pins.items()):
            path = snapshot / name
            add('evidence/' + path.relative_to(ROOT).as_posix(), path, pin)
        snapshots[mode] = {'path': str(snapshot), 'files': pins}
    proof = plan['kernel_qualification']
    for folder_key, files_key in (('folder', 'files'), ('publication', 'publication_files')):
        folder, pins = Path(proof[folder_key]), proof[files_key]
        require(inventory(folder) == pins, 'complete original numerical prerequisite and publication')
        for name, pin in sorted(pins.items()):
            path = folder / name
            add('evidence/' + path.relative_to(ROOT).as_posix(), path, pin)
    test_log = (auth['phases']['qualify']['directory'] / 'command-1.log').read_text()
    completions = re.findall(r'(?m)^\s*(\d+) passed, (\d+) warning in [^\n]+$', test_log)
    require(completions == [('216', '1')], 'original complete qualification test log')
    overview = ROOT / 'research/finite-rounded-learning-results.md'
    require(receipt['overview'] == {'path': str(overview), **descriptor(overview)},
            'renderer overview bytes match receipt')
    named = {'README.md', 'LICENSE', str(overview.relative_to(ROOT)), 'scripts/render_finite_rounded_learning.py',
             'scripts/package_finite_rounded_learning.py', *publication,
             *(str((report / name).relative_to(ROOT)) for name in outputs | {'receipt.json'})}
    committed = {}
    for name in sorted(named):
        name = safe_relative(name)
        require(name in ('README.md', 'LICENSE') or name.startswith(('research/', 'scripts/')),
                'explicit publication source/document')
        path = ROOT / name
        add('publication/' + name, path)
        committed[name] = committed_blob(path, commit)
    manifest = {
        'version': VERSION, 'study': STUDY_NAME,
        'registration': descriptor(auth['plan_path']), 'publication_commit': commit,
        'committed_publication_files': committed, 'files': members,
        'gates': auth['audit']['gates'], 'advance': auth['audit']['advance'],
        'paired_comparisons': auth['audit']['allocation_comparisons'],
        'scientific_counts': auth['summary']['counts'], 'audit_counts': auth['audit']['counts'],
        'original_phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
        'source_pins': plan['sources'], 'source_snapshots': snapshots,
        'kernel_qualification': plan['kernel_qualification'],
        'engineering_attempts': [{'registration': str(qualification_plan), 'status': 'PASS',
            'tests_passed': 216, 'warnings': 1}],
        'boundary_transition_diagnostics': auth['audit']['allocation']['boundary_transition_diagnostics'],
        'scope': 'Every authenticated study and numerical-prerequisite artifact, all 63 pinned sources with both complete freeze-time snapshots, 27 model and 27 optimizer checkpoints, training orders, histories, predictions, timings, update traces, original phase records and named publication files. The rounded primitive proof and its publication are included by their registered inventories. Previous balanced-learning failure evidence remains separately published historical context. No directory is included without an explicit roster.',
        'member_mapping': 'evidence/ preserves repository-relative paths. publication/ contains explicit display assets. Absolute original paths are provenance only.',
        'manifest_coverage': 'Every tar member except MANIFEST.json, which equals this external manifest byte for byte.',
        'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, timestamps0, empty gzip filename.',
        'reproduction_limit': 'Dependencies and executable environment are not bundled. An evidence package, not an admitted scientific rerun.',
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls',
                                'optimizer_calls', 'generator_calls', 'native_calls', 'teacher_calls'), 0),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(manifest_bytes)
    archive_path = output / ASSETS[0]
    with archive_path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed, \
            tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted([*members, 'MANIFEST.json']):
            info = tarfile.TarInfo(name)
            info.size = len(manifest_bytes) if name == 'MANIFEST.json' else members[name]['bytes']
            info.mode, info.uid, info.gid, info.mtime = 0o644, 0, 0, 0
            info.uname, info.gname = '', ''
            if name == 'MANIFEST.json':
                archive.addfile(info, io.BytesIO(manifest_bytes))
            else:
                with origins[name].open('rb') as stream:
                    archive.addfile(info, stream)
    with tarfile.open(archive_path, 'r:gz') as archive:
        entries = archive.getmembers()
        require(len(entries) == len(members) + 1
                and {entry.name for entry in entries} == set(members) | {'MANIFEST.json'},
                'exact archive member roster')
        for entry in entries:
            require(entry.isfile() and entry.mode == 0o644
                    and entry.uid == entry.gid == entry.mtime == 0, 'regular deterministic member')
            stream, digest = archive.extractfile(entry), hashlib.sha256()
            for block in iter(lambda stream=stream: stream.read(1024**2), b''):
                digest.update(block)
            pin = {'sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'bytes': len(manifest_bytes)} \
                if entry.name == 'MANIFEST.json' else members[entry.name]
            require(entry.size == pin['bytes'] and digest.hexdigest() == pin['sha256'],
                    'complete opaque-byte roundtrip: ' + entry.name)
    require(all(descriptor(path) == {key: members[name][key] for key in ('sha256', 'bytes')}
                for name, path in origins.items()), 'all packaged sources unchanged')
    require(authenticate(study) == auth, 'all original evidence unchanged after packaging')
    require(all(inventory(Path(value['path'])) == value['files'] for value in snapshots.values()),
            'complete source snapshots unchanged after packaging')
    require(inventory(Path(proof['folder'])) == proof['files']
            and inventory(Path(proof['publication'])) == proof['publication_files'],
            'complete prerequisite proof and publication unchanged after packaging')
    result = {'version': VERSION, 'status': 'PASS', 'publication_commit': commit,
              'archive': descriptor(archive_path), 'manifest': descriptor(output / 'manifest.json'),
              'members_checked': len(members) + 1, 'opaque_byte_roundtrip': True,
              'sources_unchanged': True, 'committed_publication_files_checked': len(committed),
              'counts': manifest['counts']}
    with (output / 'package-receipt.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (output / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(descriptor(output / name)['sha256'] + '  ' + name + '\n' for name in ASSETS[:3]))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--publication', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.report, args.output,
                             registration_sha256=args.registration_sha256,
                             commit=args.commit, publication=args.publication), sort_keys=True))


if __name__ == '__main__':
    main()
