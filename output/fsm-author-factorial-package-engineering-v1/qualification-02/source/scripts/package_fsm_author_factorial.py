"""Package a closed factorial study as a child of the published NL-LFR release.

No numerical arrays are decoded. Parent payloads are dependencies, not copied
again. Admission may hash opaque parent inputs through the held audit API.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_URL = 'https://github.com/kw2828/OpenJev/releases/tag/fsm-author-nllfr-study-v1'
PARENT_MANIFEST = 'research/fsm-author-nllfr-results/evidence-manifest.json'
PARENT_RECEIPT = 'research/fsm-author-nllfr-results/evidence-receipt.json'
PARENT_MANIFEST_SHA = '9a1b069fa6deb1f3fb417959782c8d6f7008da18e199d390ec8ea3eeef9efbe1'
PARENT_ARCHIVE_SHA = 'a8b758b58c9671e0a4c581a123b1cb5c7bce680bbad9e3976cc92bbb5e44226f'
AUDITOR = 'scripts/audit_fsm_author_factorial.py'
HELD = {
    AUDITOR: '450aeb0112d8931bd92527ae1d40eb55161b3fe2f0b65a20dda628fa49da248b',
    'tests/test_audit_fsm_author_factorial.py': '15923277d7853df840f65ee08d95f94f6c01a4c41583472aefeafbe829f0087a',
    'scripts/fsm_nllfr_budget_audit_math.py': 'd4454df6f8854d1895d7edfee75dd1e36a58ec616fb12537a7424359a538f8c6',
    'scripts/audit_fsm_author_nllfr.py': '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
}
ENGINEERING = ('output/fsm-author-nllfr-budget-engineering-v1',
               'output/fsm-author-budget-factorial-engineering-v1')
OPTIONAL_ENGINEERING = ('output/fsm-author-factorial-plot-engineering-v1',
                        'output/fsm-author-factorial-package-engineering-v1')
DOCUMENTS = ('research/fsm-author-nllfr-budget-fit-results.md',
             'research/fsm-author-nllfr-factorial-results.md',
             'research/fsm-author-budget-factorial-qualification.md',
             'research/fsm-author-factorial-harness-qualification.md',
             'research/fsm-author-publication-correction.md')
PUBLIC = 'research/fsm-author-nllfr-factorial-results'
PUBLIC_FILES = ('README.md', 'table.md', 'benchmark.png', 'benchmark.pdf',
                'plotted-values.json', 'receipt.json', 'audit.json', 'summary.json')
PLOT_FILES = ('benchmark.png', 'benchmark.pdf', 'plotted-values.json')
LICENSE = 'research/fsm-linear-controls-results/DATA_LICENSE.txt'
TRANSIENT = {'.pytest_cache', '.ruff_cache', '__pycache__'}
FORBIDDEN = {'.git', '.venv', 'venv', 'vendor', 'examples', 'node_modules', *TRANSIENT}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def regular(path):
    """Check path components before reading bytes, including forbidden data names."""
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and '..' not in path.parts, 'outside repository evidence')
    relative = path.relative_to(ROOT)
    require(not FORBIDDEN.intersection(relative.parts), 'excluded evidence tree: '+str(relative))
    name = path.name.lower()
    require(name != 'combined_data.npz' and path.suffix.lower() != '.npy'
            and not (path.suffix.lower() in ('.npz', '.zip') and ('300mv' in name or '_test' in name)),
            'excluded raw/reserved measurement file: '+str(relative))
    require(not any(parent.is_symlink() for parent in (path, *path.parents)), 'symlink evidence')
    return path


def descriptor(path):
    path = regular(path)
    require(path.is_file(), 'regular evidence file required: '+str(path))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def bound(value):
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
            and descriptor(value['path']) == value, 'evidence descriptor drift')
    return Path(value['path'])


def authenticate(study, audit_path, closure_path):
    """Closed original audit + held metadata API; never call numerical audit()."""
    closure, result = read(closure_path), read(audit_path)
    require(closure['state'] == 'EXITED' and closure['observed_exit_code'] == 0
            and closure['success'] is True and closure['audit_status'] == 'PASS'
            and closure['agreement'] is True and closure['sources_unchanged'] is True
            and closure['inputs_unchanged'] is True and closure['error'] is None
            and closure['closure_error'] is None and closure['evidence_errors'] == {}, 'successful original audit required')
    require(closure['audit_output'] == descriptor(audit_path)
            and result['status'] == 'PASS' and result['agreement'] is True
            and result['study'] == str(study)
            and result['scientific_status'] == closure['scientific_status']
            and result['scientific_status'] in ('REFERENCE_COMPLETE', 'REFERENCE_INCOMPLETE'), 'audit output identity')
    require(closure['sources_before'] == closure['sources_after']
            and set(closure['sources_before']) == set(HELD)
            and closure['inputs_before'] == closure['inputs_after'], 'original audit identity closure')
    pins = {'audit': descriptor(audit_path), 'audit_process': descriptor(closure_path)}
    for name, digest in HELD.items():
        value = closure['sources_before'][name]
        require(bound(value) == ROOT/name and value['sha256'] == digest, 'held independent audit source')
    for value in closure['inputs_before'].values():
        bound(value)
    for key in ('helper', 'log', 'qualification', 'registration', 'freeze', 'evaluation_process'):
        pins[key] = closure[key]
        bound(pins[key])
    process, freeze = Path(pins['evaluation_process']['path']), Path(pins['freeze']['path'])
    expected = ['.venv/bin/python', '-u', AUDITOR, '--study', str(study.relative_to(ROOT)),
                '--process', str(process.relative_to(ROOT)), '--freeze', str(freeze.relative_to(ROOT)),
                '--output', str(audit_path.relative_to(ROOT))]
    require(closure['command'] == expected and closure['timeout_seconds'] == 3600
            and closure['rss_cap_bytes'] == 32*1024**3, 'original audit command/caps')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('factorial_package_admission', ROOT/AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, inputs, paths, _terminal, _parent, _prior = module.authenticate(study, process, freeze)
    require(inputs == result['inputs'] == closure['admission_before'] == closure['admission_after'],
            'complete audited admission tree')
    require(all(inputs[k] == pins[k] for k in ('registration', 'freeze', 'qualification'))
            and inputs['process'] == pins['evaluation_process'], 'canonical audit roles')
    return plan, paths, result, pins


def tree(folder, *, omit_transient=False):
    folder = regular(folder)
    require(folder.is_dir(), 'required evidence directory: '+str(folder))
    found, excluded = [], []
    for directory, dirs, names in os.walk(folder, followlinks=False):
        dirs.sort()
        for name in dirs[:]:
            path = Path(directory)/name
            require(not path.is_symlink(), 'symlink directory')
            if omit_transient and name in TRANSIENT:
                dirs.remove(name)
                excluded.append(str(path.relative_to(ROOT)))
            else:
                regular(path)
        for name in sorted(names):
            path = Path(directory)/name
            if omit_transient and path.suffix == '.pyc':
                require(not path.is_symlink(), 'symlink cache file')
                excluded.append(str(path.relative_to(ROOT)))
            else:
                found.append(regular(path))
    return found, excluded


def short(value):
    return {key: value[key] for key in ('bytes', 'sha256')}


def gather(study, audit_path, closure_path, plots):
    plan, paths, result, admission = authenticate(study, audit_path, closure_path)
    parent_manifest, parent_receipt = read(ROOT/PARENT_MANIFEST), read(ROOT/PARENT_RECEIPT)
    require(descriptor(ROOT/PARENT_MANIFEST)['sha256'] == PARENT_MANIFEST_SHA
            and parent_receipt['status'] == 'PASS' and parent_receipt['manifest_sha256'] == PARENT_MANIFEST_SHA
            and parent_receipt['archive_sha256'] == PARENT_ARCHIVE_SHA, 'pinned published parent dependency')
    raw_digest = read(paths['old_registration'])['data_sha256']
    payload, inherited, omitted = {}, {}, []

    def add(path, expected=None):
        value = descriptor(path)
        if expected is not None:
            require(short(value) == expected, 'audited/copied file changed: '+str(path))
        require(value['sha256'] != raw_digest, 'raw combined archive content forbidden')
        name = str(Path(value['path']).relative_to(ROOT))
        require(name not in payload or payload[name] == short(value), 'archive path collision')
        payload[name] = short(value)

    new_audit = read(paths['new_audit'])
    require(new_audit['status'] == 'PASS' and new_audit['agreement'] is True
            and new_audit['inputs'] == result['inputs']['parents']['new'], 'audited new-budget parent')
    for directory, expected in ((study, result['inputs']['files']),
                                 (Path(new_audit['study']), new_audit['inputs']['files'])):
        files = tree(directory)[0] if directory.exists() else []
        require({str(p.relative_to(directory)) for p in files} == set(expected), 'exact audited study inventory')
        for path in files:
            add(path, expected[str(path.relative_to(directory))])
    for name in ENGINEERING:
        files, skipped = tree(ROOT/name, omit_transient=True)
        omitted.extend(skipped)
        for path in files:
            add(path)
    for name in OPTIONAL_ENGINEERING:
        if (ROOT/name).exists():
            files, skipped = tree(ROOT/name, omit_transient=True)
            omitted.extend(skipped)
            for path in files:
                add(path)
        else:
            omitted.append(name)
    for path in (audit_path, paths['new_audit'], ROOT/PARENT_MANIFEST, ROOT/PARENT_RECEIPT,
                 ROOT/LICENSE, Path(__file__), *(ROOT/name for name in HELD),
                 *(Path(value['path']) for value in admission.values())):
        add(path)
    budget_plan = read(paths['new_registration'])
    for registered in (plan, budget_plan):
        for name, digest in registered['source_sha256'].items():
            add(ROOT/name)
            require(payload[name]['sha256'] == digest, 'registered source identity')
        for item in registered['prerequisites'].values():
            name = item['path']
            path = regular(ROOT/name)
            if name in payload:
                require(payload[name]['sha256'] == item['sha256'], 'copied prerequisite identity')
            elif name in parent_manifest['files']:
                expected = parent_manifest['files'][name]
                require(expected['sha256'] == item['sha256'], 'parent release prerequisite identity')
                inherited[name] = expected
            else:
                add(path)
                require(payload[name]['sha256'] == item['sha256'], 'local prerequisite identity')
    for name in DOCUMENTS:
        if (ROOT/name).exists():
            add(ROOT/name)
        else:
            omitted.append(name)
    if plots.exists():
        from plot_fsm_author_factorial import authenticate as plot_admission

        _, plot_pins = plot_admission(study, audit_path, closure_path)
        receipt = read(plots/'receipt.json')
        require(receipt['inputs'] == plot_pins and receipt['study'] == str(study)
                and receipt['scientific_status'] == result['scientific_status']
                and set(receipt['outputs']) == set(PLOT_FILES), 'plot provenance/roster')
        require(bound(receipt['renderer']) == ROOT/'scripts/plot_fsm_author_factorial.py', 'plot renderer role')
        add(bound(receipt['renderer'])); add(plots/'receipt.json')
        for name, value in receipt['outputs'].items():
            require(bound(value) == plots/name, 'plot output role')
            add(plots/name)
    else:
        omitted.append(str(plots.relative_to(ROOT)))
    for name in PUBLIC_FILES:
        path = ROOT/PUBLIC/name
        if path.exists():
            add(path)
            source = {'audit.json': audit_path, 'summary.json': study/'summary.json',
                      **{n: plots/n for n in (*PLOT_FILES, 'receipt.json')}}.get(name)
            if source is not None:
                require(short(descriptor(source)) == payload[str(path.relative_to(ROOT))], 'public exact copy drift')
    require(authenticate(study, audit_path, closure_path)[3] == admission, 'audited inputs changed during enumeration')
    return {'scientific_status': result['scientific_status'], 'matrix_status': result['results']['matrix_status'],
        'parent_dependency': {'release': PARENT_URL, 'archive_sha256': PARENT_ARCHIVE_SHA,
            'manifest_sha256': PARENT_MANIFEST_SHA, 'required_direct_files': inherited,
            'instructions': 'Extract the pinned parent release at the same repository root before this child archive. '
                            'It supplies inherited forecasts, qualifications and upstream source/license; no runtime is bundled.'},
        'scope': 'Exact audited new-budget FIT and factorial evidence, closed original receipts, registered sources and '
                 'explicit engineering attempts. Audit PASS does not change scientific completion. Derived FIT and exposed-DEV '
                 'arrays are included; raw combined archives, reserved/example arrays, vendor trees and runtimes are excluded.',
        'admission': admission, 'files': dict(sorted(payload.items())), 'omitted_optional_or_transient': sorted(omitted),
        'payload_files': len(payload), 'payload_bytes': sum(p['bytes'] for p in payload.values())}


def archive(manifest, destination):
    """Deterministic gzip/tar, then verify every byte without extracting members."""
    destination.mkdir(parents=True, exist_ok=False)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    (destination/'evidence-manifest.json').write_bytes(manifest_bytes)
    path = destination/'evidence.tar.gz'
    with (path.open('xb') as raw,
          gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped,
          tarfile.open(fileobj=zipped, mode='w') as tar):
        for name, expected in sorted(manifest['files'].items()):
            data = regular(ROOT/name).read_bytes()
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'],
                    'file changed during archive: '+name)
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo('evidence-manifest.json')
        info.size, info.mtime, info.mode = len(manifest_bytes), 0, 0o644
        tar.addfile(info, io.BytesIO(manifest_bytes))
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        require(all(m.isfile() for m in members) and len(members) == len(manifest['files'])+1
                and {m.name for m in members} == {*manifest['files'], 'evidence-manifest.json'}, 'archive exact roster')
        require(tar.extractfile('evidence-manifest.json').read() == manifest_bytes, 'archive manifest bytes')
        for name, expected in manifest['files'].items():
            data = tar.extractfile(name).read()
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'],
                    'archive roundtrip bytes: '+name)
    return descriptor(path)


def run(study, audit_path, closure_path, plots, output):
    study, audit_path, closure_path, plots, output = [Path(p).absolute() for p in
                                                   (study, audit_path, closure_path, plots, output)]
    regular(output)
    require(not output.exists(), 'exclusive publication output')
    for path in (study, audit_path, closure_path, plots, *(ROOT/n for n in (*ENGINEERING, *OPTIONAL_ENGINEERING))):
        require(not output.is_relative_to(path) and not path.is_relative_to(output), 'publication overlaps evidence')
    manifest = gather(study, audit_path, closure_path, plots)
    require(not any((ROOT/name).is_relative_to(output) or output.is_relative_to(ROOT/name)
                    for name in manifest['files']), 'publication overlaps an input file')
    packed = archive(manifest, output)
    require(gather(study, audit_path, closure_path, plots) == manifest, 'evidence changed during packaging')
    receipt = {'status': 'PASS', 'scientific_status': manifest['scientific_status'],
        'matrix_status': manifest['matrix_status'], 'parent_release': PARENT_URL,
        'archive': packed, 'manifest': descriptor(output/'evidence-manifest.json'),
        'payload_files': manifest['payload_files'], 'payload_bytes': manifest['payload_bytes'],
        'renderer_or_model_calls': 0, 'array_decodes': 0, 'packager': descriptor(__file__),
        'data_attribution': 'FSM, Merijn Floren, KU Leuven; CC BY 4.0. License included.',
        'author_code_license': 'GPL-3.0-or-later; upstream source/license in required parent release.'}
    with (output/'evidence-receipt.json').open('x') as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'output/fsm-author-nllfr-factorial-study-v1')
    parser.add_argument('--audit', type=Path, default=ROOT/'output/fsm-author-nllfr-factorial-audit-v1/audit.json')
    parser.add_argument('--audit-process', type=Path, default=ROOT/'output/fsm-author-budget-factorial-engineering-v1/audit-process.json')
    parser.add_argument('--plots', type=Path, default=ROOT/'output/fsm-author-nllfr-factorial-plots-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'output/fsm-author-nllfr-factorial-publication-v1')
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.audit, args.audit_process, args.plots, args.output), indent=2))
