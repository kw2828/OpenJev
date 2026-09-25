"""Opaque evidence delivery after a closed FSM linear-controls study and agreeing audit.

No numerical libraries, model imports, array decoding, fitting or replay.
Source measurements are prohibited, including a renamed original archive.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = 'research/fsm-linear-controls-registration.json'
REGISTRATION_SHA256 = '996e492b7db79754fec9ac79239563431076673fe378ca696d240ca07d53c364'
LICENSE = 'research/fsm-residual-results/DATA_LICENSE.txt'
LICENSE_SHA256 = 'a1227b74fe74c9595e030fe6560874f3258d46a308943a44468be4fac3332b14'
ENGINEERING = ('output/fsm-linear-controls-engineering-v1',
               'output/fsm-linear-controls-audit-engineering-v1',
               'output/fsm-linear-controls-plot-engineering-v1',
               'output/fsm-linear-controls-package-engineering-v1')
EXTRAS = (REGISTRATION, 'research/fsm-linear-controls-protocol.md',
          'research/fsm-residual-prior-art.md',
          'research/fsm-author-reference-qualification.md', 'scripts/audit_fsm_linear_controls.py',
          'scripts/audit_fsm_correction.py', 'scripts/plot_fsm_linear_controls.py',
          'scripts/package_fsm_linear_controls.py', 'tests/test_audit_fsm_linear_controls.py')
SOURCES = {f'src/openjev/research/{name}.py' for name in
           ('fsm_linear_controls_study', 'fsm_affine_fold', 'fsm_residual_study',
            'fsm_residual', 'fsm_linear', 'fsm_data', 'fsm_study', 'fsm_gru',
            'predictive_state_correction')}
COMMAND = ['.venv/bin/python', '-u', '-m', 'openjev.research.fsm_linear_controls_study',
           '--protocol', REGISTRATION, '--data',
           'output/fsm-correction-engineering-v1/admission-01/combined_data.npz',
           '--output', 'output/fsm-linear-controls-study-v1']
AUDIT_COMMAND = ['.venv/bin/python', 'scripts/audit_fsm_linear_controls.py', '--study',
                 'output/fsm-linear-controls-study-v1', '--process',
                 'output/fsm-linear-controls-engineering-v1/original-process.json', '--output',
                 'output/fsm-linear-controls-audit-v1/audit.json']
COMMON_SHA256 = '2b38af454f28eb5c606bd3cadc4698dcdbcd5996e629dbb05a15371f0b440c8b'
FORBIDDEN_NAMES = {'combined_data.npz', 'raw', 'raw-data', 'raw_data', 'measurements'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def pin(path):
    data = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def regular(path, root):
    """Reject traversal, symlink ancestors and nonregular filesystem objects."""
    path, root = Path(path).absolute(), Path(root).resolve()
    require('..' not in path.parts and path.is_relative_to(root), 'path outside repository')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'symlink prohibited: '+str(item))
        if item == root:
            break
    require(path.exists(), 'missing evidence: '+str(path))
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'nonregular evidence')
    return path


def tree(folder, root):
    folder = regular(folder, root)
    require(folder.is_dir(), 'evidence directory required')
    files = []
    for path in sorted(folder.rglob('*')):
        path = regular(path, root)
        if path.is_file():
            files.append(path)
    return files


def inventory(folder, root):
    return {p.relative_to(folder).as_posix(): pin(p) for p in tree(folder, root)}


def descriptor_matches(value, path):
    return (isinstance(value, dict) and value.get('path') == str(Path(path).resolve())
            and {k: value.get(k) for k in ('sha256', 'bytes')} == pin(path))


def validate_counts(counts):
    """Check the auditor's declared roster, allowing genuine missing/failed banks."""
    fixed = {'new_linear_fits': 9, 'frozen_checkpoints': 12,
             'new_neural_updates': 0, 'fold_attempts': 3,
             'evaluation_slots': 25, 'record_rows': 300}
    variable = {'saved_prediction_banks': 300, 'parity_banks': 36,
                'parity_comparisons': 36, 'timing_samples': 600,
                'statistics_orders': 3}
    require(isinstance(counts, dict) and set(counts) == {*fixed, *variable},
            'audit count roster')
    require(all(type(counts[k]) is int and counts[k] == v for k, v in fixed.items()),
            'audit fixed counts')
    require(all(type(counts[k]) is int and 0 <= counts[k] <= v
                for k, v in variable.items()), 'audit dynamic counts')
    require(counts['parity_banks'] == counts['parity_comparisons'],
            'saved parity bank comparison coverage')
    return counts.copy()


def authenticate(study, audit_path, plots, root=ROOT):
    """Validate metadata and opaque byte identities before packaging anything."""
    root = Path(root).resolve()
    study, audit_path, plots = (regular(p, root) for p in (study, audit_path, plots))
    require(study == root/'output/fsm-linear-controls-study-v1', 'exact original study required')
    require(plots == root/'output/fsm-linear-controls-plots-v1', 'exact plot directory required')
    registration = root/REGISTRATION
    require(pin(regular(registration, root))['sha256'] == REGISTRATION_SHA256,
            'registration changed')
    plan = read(registration)
    require(read(study/'protocol.json') == plan, 'study registration copy differs')
    require(set(plan['source_sha256']) == SOURCES, 'scientific source roster')
    for name, expected in plan['source_sha256'].items():
        require(pin(regular(root/name, root))['sha256'] == expected
                == pin(regular(study/'source'/name, root))['sha256'], 'scientific source changed')
    require(plan['protocol_markdown_path'] == 'research/fsm-linear-controls-protocol.md'
            and pin(root/plan['protocol_markdown_path'])['sha256'] == plan['protocol_markdown_sha256'],
            'frozen protocol changed')
    closure = read(study/'closure.json')
    require(closure['status'] == 'completed', 'original study must be terminal')
    audit = read(audit_path)
    require(audit['status'] == 'PASS' and audit['agreement'] is True,
            'independent scientific agreement required')
    counts = validate_counts(audit['counts'])
    require(audit['inputs']['study'] == str(study)
            and audit['inputs']['registration'] == pin(registration)
            and audit['source_pins'] == plan['source_sha256'], 'audit study/source join')
    require(audit['inputs']['files'] == inventory(study, root), 'audited study inventory changed')
    require(audit['auditor'] == pin(regular(root/'scripts/audit_fsm_linear_controls.py', root))
            and pin(regular(root/'scripts/audit_fsm_correction.py', root))['sha256'] == COMMON_SHA256,
            'auditor source identity changed')
    require(audit_path == root/AUDIT_COMMAND[-1], 'original audit destination required')
    audit_process = regular(root/ENGINEERING[0]/'audit-process.json', root)
    audit_terminal = read(audit_process)
    require(type(audit_terminal['observed_exit_code']) is int and audit_terminal['observed_exit_code'] == 0
            and audit_terminal['command'] == AUDIT_COMMAND
            and type(audit_terminal['tool_session_id']) is int and audit_terminal['tool_session_id'] > 0
            and audit_terminal['observation'] == 'original Codex exec/write_stdin completion'
            and audit_terminal['audit_sha256'] == pin(audit_path)['sha256'],
            'successful original audit process/output join required')
    process = regular(Path(audit['inputs']['process']['path']), root)
    require(process == root/ENGINEERING[0]/'original-process.json'
            and descriptor_matches(audit['inputs']['process'], process), 'original process pin')
    terminal = read(process)
    require(type(terminal['observed_exit_code']) is int and terminal['observed_exit_code'] == 0
            and terminal['command'] == COMMAND
            and type(terminal['tool_session_id']) is int and terminal['tool_session_id'] > 0
            and terminal['observation'] == 'original Codex exec/write_stdin completion',
            'successful original process required')
    require(closure['source_sha256'] == plan['source_sha256']
            and closure['summary_sha256'] == terminal['summary_sha256']
            == pin(study/'summary.json')['sha256'], 'original terminal summary join')
    summary = read(study/'summary.json')
    require(closure['result'] == summary['status'] == audit['scientific_status']
            == audit['results']['status'] and summary['status'] in ('DEVELOPMENT_PASS', 'DEVELOPMENT_FAIL')
            and summary['conditions'] == audit['results']['conditions']
            and summary['passed'] == audit['results']['passed']
            and summary['total'] == audit['results']['total'] == 9,
            'saved scientific decision disagreement')
    plot = read(plots/'plot-receipt.json')
    require(plot['study'] == str(study)
            and set(plot['inputs']) == {'summary.json', 'evaluations.json'}, 'plot study identity')
    for name, item in plot['inputs'].items():
        require(descriptor_matches(item, study/name), 'plot input changed')
    require(descriptor_matches(plot['renderer'], root/'scripts/plot_fsm_linear_controls.py'),
            'renderer source changed')
    require(set(plot['outputs']) == {'benchmark.png', 'benchmark.pdf', 'plotted-values.json'}
            and set(inventory(plots, root)) == {'plot-receipt.json', *plot['outputs']},
            'plot output roster')
    for name, item in plot['outputs'].items():
        require(Path(name).name == name and descriptor_matches(item, plots/name), 'plot output changed')
    license_path = regular(root/LICENSE, root)
    require(pin(license_path)['sha256'] == LICENSE_SHA256, 'original CC BY 4.0 license changed')
    return {'scientific_status': summary['status'], 'passed': summary['passed'], 'total': 9,
            'registration': pin(registration), 'audit': pin(audit_path),
            'original_process': {'path': str(process.relative_to(root)), **pin(process)},
            'audit_process': {'path': str(audit_process.relative_to(root)), **pin(audit_process)},
            'raw_measurement_sha256': plan['data_sha256'], 'license': pin(license_path),
            'audited_counts': counts}


def collect(roots, extra_files, *, root, study, forbidden_sha256):
    """Collect every declared file, rejecting raw data instead of silently omitting it."""
    root, study = Path(root).resolve(), Path(study).absolute()
    files = {path for folder in roots for path in tree(folder, root)}
    files.update(regular(path, root) for path in extra_files)
    result = {}
    for path in sorted(files):
        require(path.is_file(), 'file required')
        relative = path.relative_to(root).as_posix()
        require(not ({part.lower() for part in path.parts} & FORBIDDEN_NAMES)
                and path.suffix.lower() not in ('.mat', '.csv', '.npy'), 'raw measurement file prohibited')
        require(path.suffix.lower() != '.npz' or path.is_relative_to(study),
                'NPZ outside independently audited study prohibited')
        require(pin(path)['sha256'] != forbidden_sha256, 'renamed raw measurement archive prohibited')
        result[relative] = path
    return result


def archive_payloads(files, output, metadata):
    """Write, fully round-trip and rehash all payload bytes; never extract files."""
    output = Path(output)
    require(not output.exists(), 'exclusive output required')
    for name in files:
        parts = PurePosixPath(name).parts
        require(parts and not PurePosixPath(name).is_absolute() and '..' not in parts
                and '\\' not in name and str(PurePosixPath(name)) == name
                and name != 'manifest.json', 'unsafe archive name')
    entries = [{'path': name, **pin(path)} for name, path in sorted(files.items())]
    manifest = {**metadata, 'file_count': len(entries),
                'payload_bytes': sum(item['bytes'] for item in entries), 'files': entries}
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output/'manifest.json'
    write(manifest_path, manifest)
    archive = output/'evidence.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for item in entries:
            data = files[item['path']].read_bytes()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                    == {k: item[k] for k in ('sha256', 'bytes')}, 'source changed during packaging')
            info = tarfile.TarInfo('evidence/'+item['path'])
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
        data = manifest_path.read_bytes()
        info = tarfile.TarInfo('evidence/manifest.json')
        info.size, info.mode = len(data), 0o644
        tar.addfile(info, io.BytesIO(data))
    expected = {'evidence/'+e['path']: {k: e[k] for k in ('sha256', 'bytes')} for e in entries}
    expected['evidence/manifest.json'] = pin(manifest_path)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(expected) and {m.name for m in members} == set(expected),
                'archive member roster differs')
        for member in members:
            require(member.isfile(), 'nonregular archive member')
            stream = tar.extractfile(member)
            require(stream is not None, 'unreadable archived payload')
            data = stream.read()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                    == expected[member.name], 'archived bytes differ')
    for item in entries:
        require(pin(files[item['path']]) == {k: item[k] for k in ('sha256', 'bytes')},
                'original changed after packaging')
    return {'archive': {'path': str(archive), **pin(archive)}, 'manifest': pin(manifest_path),
            'file_count': len(entries), 'archive_members': len(expected),
            'payload_bytes': manifest['payload_bytes'], 'all_archived_payloads_verified': True,
            'all_original_payloads_unchanged': True}


def package(study, audit_path, plots, output, *, root=ROOT):
    root = Path(root).resolve()
    study, audit_path, plots, output = (Path(p).absolute() for p in (study, audit_path, plots, output))
    require(not output.exists() and '..' not in output.parts and output.is_relative_to(root),
            'exclusive output inside repository required')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'symlink output prohibited')
    roots = [study, *(root/name for name in ENGINEERING), audit_path.parent, plots]
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots),
            'input/output overlap prohibited')
    admission = authenticate(study, audit_path, plots, root)
    extras = [root/name for name in EXTRAS] + [root/LICENSE]
    plan = read(root/REGISTRATION)
    extras.extend(root/name for name in plan['source_sha256'])
    files = collect(roots, extras, root=root, study=study,
                    forbidden_sha256=admission['raw_measurement_sha256'])
    require('DATA_LICENSE.txt' not in files, 'license alias collision')
    files['DATA_LICENSE.txt'] = root/LICENSE
    metadata = {'scope': 'Complete original linear-controls study, all declared engineering attempts, audit and plots; opaque copies, no scientific replay.',
                'admission': admission, 'source_data_license': 'CC BY 4.0',
                'source_data_author': 'Merijn Floren, KU Leuven; Floren et al., ISMA-USD 2024',
                'source_data_url': 'https://github.com/merijnfloren/fsm-benchmark-data/tree/539a12fef384b086a8562b500498b2fa3899ef70',
                'source_data_changes': '100/200 mV estimation records; whole realization/period split, FIT normalization and C100/H128 windows. Saved derived targets and forecasts included. No 300 mV or official-test array decoding.',
                'raw_measurements': 'Source archive excluded by root selection and rejected by name/content hash if encountered; no raw measurement payloads permitted.',
                'lineage': 'Manifest paths preserve repository hierarchy. DATA_LICENSE.txt is a byte-identical alias of the pinned original license. All scientific failures and negative decisions are retained.'}
    receipt = archive_payloads(files, output, metadata)
    require(authenticate(study, audit_path, plots, root) == admission, 'post-package admission changed')
    require(collect(roots, extras, root=root, study=study,
                    forbidden_sha256=admission['raw_measurement_sha256']) ==
            {name: path for name, path in files.items() if name != 'DATA_LICENSE.txt'},
            'input inventory changed during packaging')
    (output/'DATA_LICENSE.txt').write_bytes((root/LICENSE).read_bytes())
    require(pin(output/'DATA_LICENSE.txt') == admission['license'], 'license copy differs')
    receipt.update({'admission': admission, 'packager': pin(__file__), 'status': 'PASS'})
    write(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'plots', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.audit, args.plots, args.output), indent=2))


if __name__ == '__main__':
    main()
