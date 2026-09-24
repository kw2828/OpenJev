"""Package a closed study by opaque bytes, including remapped runtime sources.

Execution requires the final closure hash after publication files are complete.
No array, checkpoint, journal content, model or simulator is loaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'otto-belief-distillation-v1'
STUDY = ROOT / 'output' / VERSION
OUT = Path(__file__).resolve().parent
NATIVE = ROOT / '.venv-otto-released-native'
REGISTRATION_PIN = 'db836bc5997a00d6f163fbcb56a4770a1ed884fe181177730e74f151676bc4c3'
ASSET_NAMES = (VERSION + '.tar.gz', 'archive-manifest.json', 'archive-verification.json', 'SHA256SUMS')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
            'regular file without symlink traversal: ' + str(path))
    require(path.is_relative_to(ROOT), 'file within original repository/runtime: ' + str(path))
    return path


def desc(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def archive_name(path):
    path = regular(path)
    if path.is_relative_to(NATIVE):
        name = 'external-runtime-sources/' + path.relative_to(NATIVE).as_posix()
    else:
        name = path.relative_to(ROOT).as_posix()
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and all(p not in ('', '.', '..') for p in parts.parts), 'safe member path')
    return name


def embedded_images(text):
    references = re.findall(r'!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))', text)
    references = [left or right for left, right in references]
    references += re.findall(r'<img\b[^>]*\bsrc=[\'"]([^\'"]+)[\'"]', text)
    paths = []
    for reference in references:
        parsed = urlsplit(reference)
        require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
                'repository-local embedded README media: ' + reference)
        relative = PurePosixPath(unquote(parsed.path))
        require(not relative.is_absolute() and '..' not in relative.parts, 'safe README media path')
        paths.append(regular(ROOT / relative))
    require(paths, 'README contains embedded media')
    return paths


def closed_phase(phase, binding, paths):
    directory = STUDY / (phase + '-01')
    receipt_path, terminal_path = directory / 'receipt.json', STUDY / (phase + '-native-01.terminal.json')
    require(desc(receipt_path) == binding['receipt'] and desc(terminal_path) == binding['terminal'], 'original phase binding')
    receipt, terminal = read(receipt_path), read(terminal_path)
    require(receipt['status'] == terminal['status'] == 'completed' and terminal['returncode'] == 0
            and not terminal['timed_out'] and terminal['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True,
            'successful original process closure')
    require(receipt['plan_sha256'] == REGISTRATION_PIN and terminal['cap_seconds'] == 1800,
            'registered phase identity and cap')
    names = set(receipt['files'])
    require({p.name for p in directory.iterdir()} == names | {'receipt.json'}, 'complete exact scientific payload roster')
    for name, expected in receipt['files'].items():
        require(Path(name).name == name and desc(directory / name) == expected, 'opaque scientific payload identity')
        paths.add(regular(directory / name))
    start = read(directory / 'started.json')
    launch = start['launch']
    require(all(terminal[k] == value for k, value in launch.items())
            and start['plan_sha256'] == REGISTRATION_PIN
            and launch['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns'] <= terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + 1800 * 10**9, 'original launch/time identity')
    command = terminal['command']
    launch_path = Path(command[command.index('--supervision') + 1])
    require(desc(launch_path) == receipt['supervision'] and read(launch_path) == launch, 'original launch bytes')
    require(command[command.index('--plan-sha256') + 1] == REGISTRATION_PIN
            and Path(command[command.index('--plan') + 1]) == STUDY / 'registration-01.json'
            and Path(command[command.index('--output') + 1]) == directory, 'original command inputs')
    paths.update((regular(receipt_path), regular(terminal_path), regular(launch_path)))
    return receipt, terminal


def main(closure_sha):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', closure_sha), 'repository root and external closure pin')
    require(all(not (OUT / name).exists() for name in ASSET_NAMES), 'exclusive new archive assets')
    registration = STUDY / 'registration-01.json'
    closure_path = STUDY / 'closure-01.json'
    require(desc(registration)['sha256'] == REGISTRATION_PIN and desc(closure_path)['sha256'] == closure_sha,
            'frozen registration and final closure')
    plan, closure = read(registration), read(closure_path)
    require(closure['version'] == VERSION and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True and closure['status'] in ('DEV_PASS', 'DEV_FAIL')
            and closure['registration'] == desc(registration)
            and closure['old_test_admitted'] is False and closure['new_execution_admitted'] is False,
            'closed development-only study')
    paths, bindings = set(), {}
    for name, expected in {**plan['sources'], **plan['inputs']}.items():
        path = regular(name)
        require(desc(path) == expected, 'unchanged registered source/input: ' + name)
        paths.add(path)
        bindings[name] = archive_name(path)
    require(set(closure['processes']) == {'collection', 'fit', 'audit'}, 'three original phase closures')
    receipts, terminals = {}, {}
    for phase, binding in closure['processes'].items():
        receipts[phase], terminals[phase] = closed_phase(phase, binding, paths)
    require(terminals['collection']['finished_ns'] <= terminals['fit']['started_ns']
            and terminals['fit']['finished_ns'] <= terminals['audit']['started_ns'], 'original phase chronology')
    # Only study top-level files and the three explicitly closed scientific
    # directories are added. Fabricated test trees and publication outputs are
    # never swept recursively into their own archive.
    paths.update(regular(p) for p in STUDY.iterdir() if p.is_file())
    required = ('README.md', 'LICENSE', 'research/experiment-index.md',
        'research/otto-belief-distillation-results.md', 'research/otto-belief-distillation-next.md',
        'research/jev-architecture-source-review.md', 'tmp/otto-source-review-01/LICENSE',
        'output/otto-belief-distillation-v1/close-01.py',
        'output/otto-belief-distillation-publication-v1/render-01.py',
        'output/otto-belief-distillation-publication-v1/render-01.executed.py',
        'output/otto-belief-distillation-publication-v1/render-01.report.md',
        'output/otto-belief-distillation-publication-v1/render-01.receipt.json',
        'output/otto-belief-distillation-publication-v1/render-01.log',
        'output/otto-belief-distillation-publication-v1/release-notes.md',
        'output/otto-belief-distillation-publication-v1/archive.py',
        'output/otto-belief-distillation-publication-v1/verify-remote-01.py',
        '.venv-otto-released-native/lib/python3.12/site-packages/numpy-2.5.3.dist-info/licenses/LICENSE.txt',
        '.venv-otto-released-native/lib/python3.12/site-packages/numpy-2.5.3.dist-info/licenses/numpy/random/LICENSE.md')
    paths.update(regular(name) for name in required)
    paths.update(embedded_images((ROOT / 'README.md').read_text()))
    result_dir = ROOT / 'research/otto-belief-distillation-results'
    render_receipt = read(result_dir / 'receipt.json')
    require(render_receipt['closure'] == desc(closure_path)
            and render_receipt['audit'] == receipts['audit']['files']['audit.json']
            and render_receipt['renderer'] == desc(OUT / 'render-01.py'), 'renderer binds closed audit and source')
    first_render = read(OUT / 'render-01.receipt.json')
    require(render_receipt['first_render_receipt'] == desc(OUT / 'render-01.receipt.json')
            and render_receipt['first_render_source'] == desc(OUT / 'render-01.executed.py')
            and first_render['renderer'] == desc(OUT / 'render-01.executed.py')
            and first_render['closure'] == render_receipt['closure']
            and first_render['audit'] == render_receipt['audit']
            and first_render['files']['research/otto-belief-distillation-results.md'] == desc(OUT / 'render-01.report.md'),
            'preserved initial render bytes and original audit binding')
    for name, expected in render_receipt['files'].items():
        require(desc(name) == expected, 'unchanged rendered publication')
        paths.add(regular(name))
    paths.add(regular(result_dir / 'receipt.json'))
    require({p for p in result_dir.iterdir()} <= paths, 'complete rendered result directory')
    files, by_name = {}, {}
    for path in sorted(paths):
        name = archive_name(path)
        require(name not in files and name != 'MANIFEST.json', 'unique archive member')
        files[name] = {**desc(path), 'original_path': str(path),
                       'external_runtime_source': path.is_relative_to(NATIVE)}
        by_name[name] = path
    manifest = {'study': VERSION, 'files': files, 'registered_bindings': bindings,
        'closure': desc(closure_path), 'registration': desc(registration), 'scientific_status': closure['status'],
        'scope': 'All registered source/direct input bytes, exact current-study scientific payloads/checkpoints/journals, closed process metadata and completed publication/media. Old raw TEST is not added.',
        'manifest_coverage': 'Every tar member except MANIFEST.json; that member is an identical copy of this external manifest.',
        'runtime_source_mapping': 'Absolute installed NumPy paths are mapped below external-runtime-sources/, relative to the original .venv-otto-released-native directory. Original paths remain in each record.',
        'external_dependencies': ['Qualified Python/Torch/TensorFlow environments and executables are not bundled.',
            'Direct registered native weights/tensors are bundled. Earlier authentication chains may reference receipts and resources outside these direct inputs; use the repository and previous evidence releases.',
            'Absolute paths describe the original workstation. This evidence archive is not a portable installer or an admitted rerun.',
            'Public README media from earlier studies are presentation artifacts, not newly admitted scientific data.'],
        'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, tar/gzip timestamps0.',
        'array_decodes': 0, 'checkpoint_decodes': 0, 'journal_content_parses': 0, 'model_or_native_calls': 0}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    with (OUT / 'archive-manifest.json').open('xb') as stream:
        stream.write(manifest_bytes)
    archive = OUT / ASSET_NAMES[0]
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name in sorted([*files, 'MANIFEST.json']):
            info = tarfile.TarInfo(name)
            info.size = len(manifest_bytes) if name == 'MANIFEST.json' else files[name]['bytes']
            info.mode, info.uid, info.gid, info.mtime = 0o644, 0, 0, 0
            if name == 'MANIFEST.json':
                tar.addfile(info, io.BytesIO(manifest_bytes))
            else:
                with by_name[name].open('rb') as stream:
                    tar.addfile(info, stream)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(files) + 1 and {m.name for m in members} == set(files) | {'MANIFEST.json'}, 'exact archive inventory')
        for member in members:
            require(member.isfile(), 'regular archive members only')
            stream = tar.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda stream=stream: stream.read(1024**2), b''):
                digest.update(block)
            expected = {'sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'bytes': len(manifest_bytes)} if member.name == 'MANIFEST.json' else files[member.name]
            require(member.size == expected['bytes'] and digest.hexdigest() == expected['sha256'], 'opaque archive byte roundtrip')
    require(all(desc(path) == {k: files[name][k] for k in ('sha256', 'bytes')} for name, path in by_name.items()), 'all source bytes unchanged after packaging')
    verification = {'status': 'passed', 'members_checked': len(files) + 1, 'payload_members': len(files),
        'archive': desc(archive), 'manifest': desc(OUT / 'archive-manifest.json'), 'closure': desc(closure_path),
        'opaque_byte_roundtrip': True, 'sources_unchanged': True, 'numerical_decodes': 0}
    with (OUT / 'archive-verification.json').open('x') as stream:
        json.dump(verification, stream, sort_keys=True, indent=2)
        stream.write('\n')
    with (OUT / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(desc(OUT / name)['sha256'] + '  ' + name + '\n' for name in ASSET_NAMES[:3]))
    print(json.dumps(verification))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-sha256', required=True)
    main(parser.parse_args().closure_sha256)
