"""Package authenticated closed convex-readout evidence as opaque bytes.

No data/checkpoint arrays are decoded and no model/generator is imported. The
only discovered records are explicitly named engineering registrations and
their authenticated inventories. Repository or study directories are never
added wholesale. Execution does not commit, publish or alter scientific files.
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

from render_finite_gap_readout_study import (
    ROOT,
    authenticate,
    descriptor,
    inventory,
    read,
    regular,
    require,
)

VERSION = 'finite-gap-readout-study-package-v1'
STUDY_NAME = 'finite-gap-readout-study-v1'
ASSETS = (STUDY_NAME + '.tar.gz', 'manifest.json', 'package-receipt.json', 'SHA256SUMS')


def safe_relative(name):
    require(type(name) is str and name and '\\' not in name, 'plain portable member name')
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and all(part not in ('', '.', '..') for part in name.split('/')),
            'safe relative archive member')
    return parts.as_posix()


def committed_blob(path, commit):
    name = regular(path).relative_to(ROOT).as_posix()
    entries = subprocess.check_output(['git', 'ls-tree', '-z', commit, '--', name], cwd=ROOT, timeout=30).split(b'\0')
    entries = [entry for entry in entries if entry]
    require(len(entries) == 1, 'single committed publication file: ' + name)
    header, saved_name = entries[0].split(b'\t', 1)
    mode, kind, blob = header.decode().split()
    require(saved_name.decode() == name and kind == 'blob' and mode in ('100644', '100755'), 'regular committed publication blob')
    require(subprocess.check_output(['git', 'cat-file', 'blob', blob], cwd=ROOT, timeout=30) == path.read_bytes(),
            'publication commit matches current file: ' + name)
    return {'git_blob': blob, **descriptor(path)}



def package(studyfolder, reportfolder, outputfolder, *, registration_sha256, commit=None, publication_paths=()):
    studyfolder, reportfolder = Path(studyfolder).resolve(), Path(reportfolder).resolve()
    outputfolder = Path(outputfolder)
    require(re.fullmatch('[0-9a-f]{64}', registration_sha256), 'external registration digest')
    require(studyfolder == ROOT / 'output' / STUDY_NAME, 'exact registered study folder')
    require(descriptor(studyfolder / 'study-registration.json')['sha256'] == registration_sha256, 'owner-supplied registration identity')
    require(outputfolder.is_absolute() and outputfolder.resolve() == outputfolder and not outputfolder.exists(),
            'exclusive absolute archive directory')
    plan_path, plan, phases, saved, audit = authenticate(studyfolder)
    if commit is not None:
        require(re.fullmatch('[0-9a-f]{40}', commit)
                and subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=ROOT, text=True, timeout=30).strip() == commit,
                'existing full publication commit')
    require(reportfolder.is_relative_to(ROOT) and reportfolder.is_dir() and not reportfolder.is_symlink(), 'local completed report folder')
    render_receipt = read(reportfolder / 'receipt.json')
    require(render_receipt['version'] == 'finite-gap-readout-study-report-v1'
            and render_receipt['registration'] == {'path': str(plan_path), **descriptor(plan_path)}
            and render_receipt['sources'] == plan['sources']
            and render_receipt['method_qualification'] == plan['method_qualification']
            and render_receipt['predecessor'] == plan['predecessor']
            and render_receipt['engineering_predecessor'] == plan['engineering_predecessor']
            and render_receipt['model_selection'] is False and render_receipt['result_reads_after_original_closure'] is True
            and all(value == 0 for value in render_receipt['counts'].values()), 'completed metadata-only renderer')
    expected_outputs = {'summary.json', 'report.md', 'benchmark.png'}
    require(set(render_receipt['files']) == expected_outputs, 'exact rendered publication outputs')
    require(inventory(reportfolder) == {**render_receipt['files'], 'receipt.json': descriptor(reportfolder / 'receipt.json')},
            'all completed report bytes match receipt')
    renderer_path = regular(ROOT / 'scripts/render_finite_gap_readout_study.py')
    require(render_receipt['renderer'] == {'path': str(renderer_path), **descriptor(renderer_path)}, 'original renderer bytes')
    for phase, record in phases.items():
        expected = {kind: {'path': str(record[kind + '_path']), **descriptor(record[kind + '_path'])}
                    for kind in ('receipt', 'launch', 'terminal')}
        require(render_receipt['inputs'][phase] == expected, 'renderer original phase joins')

    members, origins = {}, {}

    def add(name, path, expected=None):
        name, path = safe_relative(name), regular(path)
        require(path.is_relative_to(ROOT) and name != 'MANIFEST.json', 'repository-bound source and nonreserved member')
        actual = descriptor(path)
        require(expected is None or actual == expected, 'named archive source identity: ' + name)
        if name in members:
            require(origins[name] == path and members[name]['sha256'] == actual['sha256'], 'unambiguous duplicate archive binding')
            return
        members[name] = {**actual, 'original_path': str(path)}
        origins[name] = path

    def study_file(path, expected=None):
        path = regular(path)
        require(path.is_relative_to(studyfolder), 'original metadata/payload stays within this study')
        add('study/' + path.relative_to(studyfolder).as_posix(), path, expected)

    for name, pin in plan['sources'].items():
        add('sources/' + safe_relative(name), ROOT / name, pin)
    study_file(plan_path)
    upstream_folder = regular(Path(plan['upstream']['registration']['path'])).parent
    require(upstream_folder == ROOT / 'output/finite-factorized-dynamics-v1'
            and inventory(upstream_folder) == plan['upstream']['files'], 'exact unchanged upstream proof inventory')
    for name, pin in plan['upstream']['files'].items():
        name = safe_relative(name)
        add('upstream/finite-factorized-dynamics-v1/' + name, upstream_folder / name, pin)
    for key, expected_folder in (('method_qualification', 'finite-gap-solver-qualification-v1'),
                                 ('predecessor', 'finite-convex-readout-v1')):
        proof = plan[key]
        folder = ROOT / 'output' / expected_folder
        require(proof['folder'] == str(folder) and inventory(folder) == proof['files'], 'exact named historical inventory: ' + key)
        for name, pin in proof['files'].items():
            name = safe_relative(name)
            add('history/' + expected_folder + '/' + name, folder / name, pin)
    for record in phases.values():
        for name, pin in record['receipt']['files'].items():
            study_file(record['directory'] / safe_relative(name), pin)
        for kind in ('receipt_path', 'launch_path', 'terminal_path'):
            study_file(record[kind])
        study_file(record['launch_path'].with_name(record['launch_path'].name[:-len('.launch.json')] + '.log'))
    qualification_plan = regular(Path(phases['qualify']['receipt']['plan']))
    engineering_plans = sorted(studyfolder.glob('engineering-registration-[0-9][0-9].json'))
    require(qualification_plan == studyfolder / 'engineering-registration-02.json'
            and engineering_plans == [studyfolder / 'engineering-registration-01.json', qualification_plan],
            'exact failed-first and successful-second engineering roster')
    engineering_attempts = []
    for engineering_plan in engineering_plans:
        attempt_id = engineering_plan.stem.rsplit('-', 1)[-1]
        record = read(engineering_plan)
        require(record['version'] == STUDY_NAME and record['mode'] == 'engineering' and record['root'] == str(ROOT),
                'same-study engineering attempt')
        sources = record['sources']
        study_file(engineering_plan)
        snapshot = studyfolder / ('engineering-source-' + attempt_id)
        require(snapshot.exists() or engineering_plan == qualification_plan,
                'superseded qualification requires preserved source snapshot')
        if snapshot.exists():
            snapshot_files = inventory(snapshot)
            if engineering_plan != qualification_plan:
                require(snapshot == Path(plan['engineering_predecessor']['folder'])
                        and snapshot_files == plan['engineering_predecessor']['files']
                        and set(snapshot_files) == set(sources) | {'manifest.json'}
                        and {name: snapshot_files[name] for name in sources} == sources,
                        'complete failed source snapshot including its pinned manifest')
                require(read(snapshot / 'manifest.json') == {
                    'registration_sha256': descriptor(engineering_plan)['sha256'], 'sources': sources},
                    'failed source manifest joins its exact original registration')
                study_file(snapshot / 'manifest.json', snapshot_files['manifest.json'])
            else:
                require(snapshot_files == sources, 'complete qualifying engineering source snapshot')
        for name, pin in sources.items():
            source = snapshot / name if snapshot.exists() else ROOT / name
            add('study/engineering-source-' + attempt_id + '/' + safe_relative(name), source, pin)
        if engineering_plan == qualification_plan:
            receipt, terminal = phases['qualify']['receipt'], phases['qualify']['terminal']
        else:
            spec = record['phases']['qualify']
            folder, launch_path = Path(spec['output']), regular(Path(spec['supervision']))
            require(launch_path.name.endswith('.launch.json'), 'original failed-attempt launch suffix')
            terminal_path = launch_path.with_name(launch_path.name[:-len('.launch.json')] + '.terminal.json')
            receipt_path = Path(str(folder) + '.receipt.json')
            receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
            require(receipt['phase'] == 'qualify' and receipt['status'] == 'FAILED'
                    and receipt['plan'] == str(engineering_plan)
                    and receipt['plan_sha256'] == descriptor(engineering_plan)['sha256']
                    and receipt['sources_before'] == sources
                    and receipt['supervision'] == str(launch_path) and receipt['output'] == str(folder),
                    'original failed qualification identity, not scientific evidence')
            require(receipt['launch'] == launch and all(terminal[key] == value for key, value in launch.items())
                    and terminal['status'] == 'failed' and terminal['group_absent'] is True
                    and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True,
                    'closed original failed attempt and no surviving process group')
            command = terminal['command']
            expected = {'--plan': str(engineering_plan), '--plan-sha256': descriptor(engineering_plan)['sha256'],
                        '--phase': 'qualify', '--supervision': str(launch_path), '--output': str(folder)}
            require(terminal['cwd'] == str(ROOT) and command[0] == record['runtime']['executable']
                    and (ROOT / command[1]).resolve() == ROOT / 'scripts/finite_gap_readout_study_worker.py'
                    and len(command) == 12 and len(set(command[2::2])) == 5, 'failed original worker and unique flags')
            observed = dict(zip(command[2::2], command[3::2], strict=True))
            require(set(observed) == set(expected), 'failed original argument roster')
            for flag in ('--plan', '--supervision', '--output'):
                observed[flag] = str((ROOT / observed[flag]).resolve())
            require(observed == expected and terminal['cap_seconds'] == spec['cap_seconds']
                    and terminal['watchdog_sha256'] == sources['scripts/supervise_dialogue_observation_v2.py']['sha256']
                    and terminal['clock_source_sha256'] == sources['src/openjev/research/suspend_clock.py']['sha256'],
                    'source-bound failed supervision')
            require(inventory(folder) == receipt['files'], 'failed attempt inventory retained exactly')
            for name, pin in receipt['files'].items():
                study_file(folder / safe_relative(name), pin)
            for path in (receipt_path, launch_path, terminal_path,
                         launch_path.with_name(launch_path.name[:-len('.launch.json')] + '.log')):
                study_file(path)
        engineering_attempts.append({'registration': engineering_plan.name, 'status': receipt['status'],
                                     'qualifying': engineering_plan == qualification_plan,
                                     'terminal_status': terminal['status'], 'wall_seconds': terminal['wall_seconds']})
    optional_closure = studyfolder / 'closure-01.json'
    if optional_closure.exists():
        study_file(optional_closure)
    for name in (*sorted(expected_outputs), 'receipt.json'):
        add('publication/report/' + name, reportfolder / name)
    for name in ('README.md', 'LICENSE', 'scripts/render_finite_gap_readout_study.py', 'scripts/package_finite_gap_readout_study.py'):
        add('publication/' + name, ROOT / name)
    publication_failure_folder = ROOT / 'output/finite-gap-readout-publication-attempt-01'
    publication_failure_files = inventory(publication_failure_folder)
    preserved_helpers = {
        'render_finite_gap_readout_study.py': {'bytes': 50753,
            'sha256': '0fc301e371b6905514568b79e2b57fc5422d66c9df2228a4f61202d8d6429fc4'},
        'package_finite_gap_readout_study.py': {'bytes': 20696,
            'sha256': 'a9f05eeaea80453b17aaeec30b0b8d9d2623cd07cbb8d89ccf7613627e095b64'}}
    require(set(publication_failure_files) == set(preserved_helpers) | {'failure.json'}
            and {name: publication_failure_files[name] for name in preserved_helpers} == preserved_helpers,
            'exact preserved publication failure and two original helper sources')
    publication_failure = read(publication_failure_folder / 'failure.json')
    require(publication_failure == {
        'command': ['render_finite_gap_readout_study.py', '--study', 'output/finite-gap-readout-study-v1',
                    '--output', 'research/finite-gap-readout-study-results'],
        'error': 'ValueError: all original failed-attempt sources preserved separately',
        'files': preserved_helpers, 'scientific_outputs_changed': False, 'status': 'FAILED_BEFORE_METRIC_READS'},
        'original publication-only failure record joins its preserved sources')
    for name, pin in publication_failure_files.items():
        add('publication-attempts/attempt-01/' + name, publication_failure_folder / name, pin)
    publication_files = [ROOT / 'README.md', ROOT / 'LICENSE']
    for name in publication_paths:
        name = safe_relative(name)
        path = regular(ROOT / name)
        require((name.startswith('research/') or name == 'README.md')
                and path.suffix.lower() in ('.md', '.json', '.png', '.gif', '.svg'), 'explicit publication document/media only')
        add('publication/' + name, path)
        publication_files.append(path)
    committed = {path.relative_to(ROOT).as_posix(): committed_blob(path, commit)
                 for path in sorted(set(publication_files))} if commit is not None else {}
    manifest = {'version': VERSION, 'study': STUDY_NAME, 'registration': descriptor(plan_path),
        'publication_commit': commit, 'committed_publication_files': committed,
        'files': members, 'gates': audit['gates'], 'recorded_scientific_work': saved['counts'],
        'recorded_audit_work': audit['counts'],
        'certificates': audit['certificates'],
        'paired_comparisons': audit['paired_comparisons'],
        'solve_costs': [{'parent': row['parent'], 'seed': row['seed'],
                         'load_seconds': row['load_seconds'], 'extraction_seconds': row['extraction_seconds'],
                         'solve_seconds': row['solve_seconds'], 'solver_work': row['solver_result']['work']}
                        for row in saved['solves']],
        'frozen_invariants': audit['frozen_invariants'],
        'structural_work': saved['structural_work'],
        'upstream': plan['upstream'],
        'method_qualification': plan['method_qualification'], 'predecessor': plan['predecessor'],
        'engineering_predecessor': plan['engineering_predecessor'],
        'upstream_scope': 'Every explicitly pinned original factorized-dynamics member is opaque historical provenance. Original TRAIN and nine checkpoints are reused scientific inputs. Prior DEV arrays are retained only for proof closure and were not decoded or reused for new evaluation.',
        'original_phase_seconds': {name: phase['terminal']['wall_seconds'] for name, phase in phases.items()},
        'engineering_attempts': engineering_attempts,
        'publication_attempts': [{'folder': str(publication_failure_folder),
                                  'files': publication_failure_files, 'record': publication_failure}],
        'scope': 'All registered sources; all current closed producer/audit payloads including nine complete solves, stopping histories and state caches, original TRAIN copy, fresh development arrays, oracle controls, predictions and journals; original receipts/launches/terminals/logs; every named engineering attempt with source snapshots; complete explicitly pinned parent-study, numerical-qualification and failed-predecessor inventories; completed report. No new recurrent checkpoint is trained, and failed attempts remain failures.',
        'member_mapping': 'study/ preserves current study-relative names; upstream/finite-factorized-dynamics-v1/ preserves parent-study names; history/ preserves qualification and failed-predecessor names; sources/ preserves repository source paths; publication/ contains named presentation artifacts. Only explicit pins are included; absolute original paths remain provenance metadata only.',
        'manifest_coverage': 'Every tar member except MANIFEST.json; that member equals this external manifest byte for byte.',
        'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, timestamps0 and gzip filename empty.',
        'reproduction_limit': 'Dependencies and executable environments are not bundled. This is an evidence package, not a portable installer or an admitted rerun.',
        'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'generator_calls': 0, 'optimizer_calls': 0}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    outputfolder.mkdir(parents=True, exist_ok=False)
    manifest_path = outputfolder / 'manifest.json'
    with manifest_path.open('xb') as stream:
        stream.write(manifest_bytes)
    archive_path = outputfolder / ASSETS[0]
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
        require(len(entries) == len(members) + 1 and {entry.name for entry in entries} == set(members) | {'MANIFEST.json'},
                'complete exact archive inventory')
        for entry in entries:
            require(entry.isfile() and entry.mode == 0o644 and entry.uid == entry.gid == entry.mtime == 0,
                    'deterministic regular archive metadata')
            source = archive.extractfile(entry)
            digest = hashlib.sha256()
            for block in iter(lambda source=source: source.read(1024**2), b''):
                digest.update(block)
            expected = {'sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'bytes': len(manifest_bytes)} \
                if entry.name == 'MANIFEST.json' else members[entry.name]
            require(entry.size == expected['bytes'] and digest.hexdigest() == expected['sha256'], 'opaque member byte roundtrip')
    require(all(descriptor(path) == {key: members[name][key] for key in ('sha256', 'bytes')}
                for name, path in origins.items()), 'all source bytes unchanged after packaging')
    for key in ('upstream', 'method_qualification', 'predecessor', 'engineering_predecessor'):
        require(inventory(Path(plan[key]['folder'])) == plan[key]['files'],
                'complete historical inventory unchanged after packaging: ' + key)
    for phase in phases.values():
        require(inventory(phase['directory']) == phase['receipt']['files'],
                'complete successful phase unchanged after packaging')
    require(inventory(publication_failure_folder) == publication_failure_files,
            'preserved publication failure unchanged after packaging')
    receipt = {'version': VERSION, 'status': 'PASS', 'registration': descriptor(plan_path),
        'archive': descriptor(archive_path), 'manifest': descriptor(manifest_path),
        'members_checked': len(members) + 1, 'opaque_byte_roundtrip': True, 'sources_unchanged': True,
        'publication_commit': commit, 'committed_publication_files_checked': len(committed),
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls', 'optimizer_calls'), 0)}
    with (outputfolder / 'package-receipt.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (outputfolder / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(descriptor(outputfolder / name)['sha256'] + '  ' + name + '\n' for name in ASSETS[:3]))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--publication', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.report, args.output, registration_sha256=args.registration_sha256,
                             commit=args.commit, publication_paths=args.publication), sort_keys=True))


if __name__ == '__main__':
    main()
