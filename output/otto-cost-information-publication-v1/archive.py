"""Package a closed study by opaque bytes, including remapped runtime sources.

Execution requires the final closure hash and publication commit from the owner.
No array, checkpoint, journal content, model or simulator is loaded.
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
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'otto-cost-information-v1'
STUDY = ROOT / 'output' / VERSION
OUT = Path(__file__).resolve().parent
NATIVE = ROOT / '.venv-otto-released-native'
REGISTRATION_PIN = 'd0a7c9ed0155afd0e8fc3bac0cb042b51e551f3efcabcaf81c0205459bc53749'
ASSET_NAMES = (VERSION + '.tar.gz', 'archive-manifest.json', 'archive-verification.json', 'SHA256SUMS')
PHASES = {'collection': ('collect', 'collect_otto_cost_information.py'),
          'prediction': ('predict', 'run_otto_cost_information.py'),
          'audit': ('audit', 'audit_otto_cost_information.py')}
PAYLOADS = {
    'collection': {'started.json', 'parent-reference.json', 'runtime.json', 'setup.json',
        'sensor-laws.npz', 'sensor-laws.json', 'cases.npz', 'cases.jsonl', 'commitments.jsonl',
        'summary.json', 'work.jsonl.gz', 'weights.jsonl.gz', 'forwards.jsonl.gz',
        'native-draws.jsonl.gz', 'shadow.jsonl.gz', 'source-laws.jsonl.gz', 'transitions.jsonl.gz'},
    'prediction': {'started.json', 'predictions.npz', 'report.json', 'summary.json'},
    'audit': {'started.json', 'audit.json'},
}


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
    require(receipt['version'] == VERSION and receipt['phase'] == PHASES[phase][0]
            and receipt['plan_sha256'] == REGISTRATION_PIN and terminal['cap_seconds'] == 1800
            and receipt['old_test_decodes'] == receipt['astra_calls'] == 0
            and receipt['wall_seconds'] == binding['worker_seconds'],
            'registered phase identity and cap')
    names = set(receipt['files'])
    require(names == PAYLOADS[phase] and {p.name for p in directory.iterdir()} == names | {'receipt.json'},
            'complete exact scientific payload roster')
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
            and Path(command[command.index('--output') + 1]) == directory
            and Path(command[1]) == ROOT / 'scripts' / PHASES[phase][1], 'original command inputs')
    paths.update((regular(receipt_path), regular(terminal_path), regular(launch_path),
                  regular(STUDY / (phase + '-native-01.log'))))
    return receipt, terminal


def published_blob(name, commit):
    entries = subprocess.check_output(['git', 'ls-tree', '-z', commit, '--', name], cwd=ROOT, timeout=30).split(b'\0')
    entries = [entry for entry in entries if entry]
    require(len(entries) == 1, 'single committed publication file: ' + name)
    header, tree_name = entries[0].split(b'\t', 1)
    mode, kind, blob = header.decode().split()
    require(tree_name.decode() == name and kind == 'blob' and mode in ('100644', '100755'), 'regular publication blob')
    data = subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=ROOT, timeout=30)
    require(data == regular(name).read_bytes(), 'publication commit matches local bytes: ' + name)
    return {'blob': blob, **desc(name)}


def main(closure_sha, commit):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', REGISTRATION_PIN)
            and re.fullmatch('[0-9a-f]{64}', closure_sha)
            and re.fullmatch('[0-9a-f]{40}', commit), 'root, external closure pin and publication commit')
    require(subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=ROOT,
                                   text=True, timeout=30).strip() == commit, 'existing full publication commit')
    require(all(not (OUT / name).exists() for name in ASSET_NAMES), 'exclusive new archive assets')
    registration = STUDY / 'registration-01.json'
    closure_path = STUDY / 'closure-01.json'
    require(desc(registration)['sha256'] == REGISTRATION_PIN and desc(closure_path)['sha256'] == closure_sha,
            'frozen registration and final closure')
    plan, closure = read(registration), read(closure_path)
    require(closure['version'] == VERSION and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True
            and closure['status'] in ('HEADROOM_RESOLVED', 'HEADROOM_NOT_RESOLVED')
            and closure['registration'] == desc(registration)
            and closure['old_test_admitted'] is False and closure['new_execution_admitted'] is False
            and closure['new_training'] is False and closure['architecture_claim'] is False,
            'closed frozen-policy diagnostic')
    paths, bindings = set(), {}
    for name, expected in {**plan['sources'], **plan['inputs']}.items():
        path = regular(name)
        require(desc(path) == expected, 'unchanged registered source/input: ' + name)
        paths.add(path)
        bindings[name] = archive_name(path)
    reference = plan['parent_reference']
    parent = ROOT / 'output/otto-action-effect-v1'
    keys = {f'{family}__{seed}' for family in ('effect_recurrent', 'paired_recurrent', 'paired_blind', 'paired_direct')
            for seed in (330000001, 330000002, 330000003)}
    require(set(reference['checkpoints']) == keys and reference['parent_data_array_decodes'] == 0
            and closure['parent_reference'] == reference, 'twelve frozen parent policies without parent data')
    parent_checkpoints = set()
    for key, record in reference['checkpoints'].items():
        family, seed = key.split('__')
        path = regular(record['path'])
        require(path == parent / 'fit-01' / f'{family}-{seed}.npz' and path in paths
                and desc(path) == {k: record[k] for k in ('sha256', 'bytes')}, 'exact parent checkpoint binding')
        parent_checkpoints.add(path)
    require({path for path in paths if path.is_relative_to(parent) and path.suffix in ('.npz', '.npy')}
            == parent_checkpoints, 'parent TRAIN/DEV/prediction arrays excluded')
    require(set(closure['processes']) == set(PHASES), 'three original phase closures')
    receipts, terminals = {}, {}
    for phase, binding in closure['processes'].items():
        receipts[phase], terminals[phase] = closed_phase(phase, binding, paths)
    require(terminals['collection']['finished_ns'] <= terminals['prediction']['started_ns']
            and terminals['prediction']['finished_ns'] <= terminals['audit']['started_ns'], 'original phase chronology')
    audit_path = STUDY / 'audit-01/audit.json'
    audit = read(audit_path)
    require(desc(audit_path) == closure['audit'] and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['gate'] == closure['gate'] and audit['gate']['status'] == closure['status']
            and audit['inputs']['collection_receipt'] == closure['processes']['collection']['receipt']
            and audit['inputs']['prediction_receipt'] == closure['processes']['prediction']['receipt'], 'closed independent audit joins')
    # Only named metadata, registered files and exact closed inventories enter.
    # No directory sweep can introduce parent data or a partial scientific run.
    paths.update((regular(registration), regular(closure_path)))
    required = ('README.md', 'LICENSE', 'research/experiment-index.md',
        'research/otto-cost-information-results.md', 'research/otto-cost-information-protocol.md',
        'research/otto-cost-information-next.md',
        'research/jev-architecture-source-review.md', 'tmp/otto-source-review-01/LICENSE',
        'third_party/otto/LICENSE', 'third_party/otto/LICENSE-zoo', 'models/chess-candidate-v2/LICENSE',
        'output/otto-cost-information-v1/close-01.py',
        'output/otto-cost-information-publication-v1/render-01.py',
        'output/otto-cost-information-publication-v1/release-notes.md',
        'output/otto-cost-information-publication-v1/archive.py',
        'output/otto-cost-information-publication-v1/verify-remote-01.py',
        '.venv-otto-released-native/lib/python3.12/site-packages/numpy-2.5.3.dist-info/licenses/LICENSE.txt',
        '.venv-otto-released-native/lib/python3.12/site-packages/numpy-2.5.3.dist-info/licenses/numpy/random/LICENSE.md')
    paths.update(regular(name) for name in required)
    render_log = OUT / 'render-01.log'
    if render_log.exists():
        paths.add(regular(render_log))
    media = embedded_images((ROOT / 'README.md').read_text())
    require(any(path.suffix.lower() == '.gif' for path in media), 'existing README GIF remains embedded')
    paths.update(media)
    result_dir = ROOT / 'research/otto-cost-information-results'
    render_receipt = read(result_dir / 'receipt.json')
    require(render_receipt['closure'] == desc(closure_path)
            and render_receipt['audit'] == receipts['audit']['files']['audit.json']
            and render_receipt['renderer'] == desc(OUT / 'render-01.py')
            and render_receipt['collection_summary'] == receipts['collection']['files']['summary.json']
            and render_receipt['prediction_summary'] == receipts['prediction']['files']['summary.json']
            and all(render_receipt[key] == 0 for key in ('array_decodes', 'checkpoint_decodes',
                'model_calls', 'native_calls', 'teacher_calls', 'optimizer_calls')),
            'renderer binds closed audit, phase summaries and source without scientific execution')
    required_rendered = {'research/otto-cost-information-results.md',
                         'research/otto-cost-information-results/benchmark.png',
                         'research/otto-cost-information-results/summary.json'}
    require(set(render_receipt['files']) == required_rendered, 'exact completed renderer outputs')
    for name, expected in render_receipt['files'].items():
        require(desc(name) == expected, 'unchanged rendered publication')
        paths.add(regular(name))
    paths.add(regular(result_dir / 'receipt.json'))
    require({p for p in result_dir.iterdir()} <= paths, 'complete rendered result directory')
    require({path for path in paths if path.is_relative_to(parent) and path.suffix in ('.npz', '.npy')}
            == parent_checkpoints, 'publication additions preserve parent checkpoint-only array scope')
    committed_names = sorted({'README.md', 'research/experiment-index.md', 'research/jev-architecture-source-review.md',
        'research/otto-cost-information-protocol.md', 'research/otto-cost-information-next.md', *required_rendered,
        'research/otto-cost-information-results/receipt.json', *[path.relative_to(ROOT).as_posix() for path in media]})
    committed_files = {name: published_blob(name, commit) for name in committed_names}
    files, by_name = {}, {}
    for path in sorted(paths):
        name = archive_name(path)
        require(name not in files and name != 'MANIFEST.json', 'unique archive member')
        files[name] = {**desc(path), 'original_path': str(path),
                       'external_runtime_source': path.is_relative_to(NATIVE)}
        by_name[name] = path
    manifest = {'study': VERSION, 'files': files, 'registered_bindings': bindings,
        'closure': desc(closure_path), 'registration': desc(registration), 'scientific_status': closure['status'],
        'publication_commit': commit, 'committed_publication_files': committed_files,
        'scope': 'All registered source/direct input bytes including twelve parent checkpoints, exact current scientific payloads/journals, original process logs and closure, qualification failures/metadata and completed publication/media. No parent TRAIN/DEV arrays, old raw TEST or reserved confirmation arrays are added.',
        'frozen_parent_policies': reference,
        'parent_evidence_release': 'https://github.com/kw2828/OpenJev/releases/tag/otto-action-effect-v1',
        'manifest_coverage': 'Every tar member except MANIFEST.json; that member is an identical copy of this external manifest.',
        'runtime_source_mapping': 'Absolute installed NumPy paths are mapped below external-runtime-sources/, relative to the original .venv-otto-released-native directory. Original paths remain in each record.',
        'external_dependencies': ['Qualified Python/Torch/TensorFlow environments and executables are not bundled.',
            'Direct registered native weights/tensors/kernels and twelve frozen parent checkpoints are bundled. Parent closure and receipts refer to excluded parent training/development and audit payloads and older native authentication chains; use the linked parent evidence release and repository for that separate evidence. No self-contained executable reproduction is claimed.',
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
        'publication_commit': commit, 'committed_publication_files_checked': len(committed_files),
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
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    main(args.closure_sha256, args.commit)
