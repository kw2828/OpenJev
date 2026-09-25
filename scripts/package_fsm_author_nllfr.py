"""Package a closed, audited NL-LFR diagnostic without changing its status."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

from plot_fsm_author_nllfr import authenticate, descriptor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    study = Path('output/fsm-author-nllfr-study-v1')
    evaluation = Path('output/fsm-author-nllfr-evaluation-v1')
    audit = Path('output/fsm-author-nllfr-audit-v1/audit.json')
    engineering = Path('output/fsm-author-engineering-v1')
    audit_process = engineering/'nllfr-audit-process.json'
    reference = Path('output/fsm-linear-controls-study-v1')
    bla = Path('output/fsm-author-bla-study-v1')
    args = tuple(p.resolve() for p in (study, audit, audit_process, evaluation, reference, bla))
    _, pins = authenticate(*args)
    audited = json.loads(audit.read_text())
    scientific = audited['scientific_status']
    if scientific not in ('REFERENCE_COMPLETE', 'REFERENCE_INCOMPLETE'):
        raise ValueError('unrecognized scientific status')
    if json.loads((evaluation/'summary.json').read_text())['status'] != scientific:
        raise ValueError('scientific status differs from audit')
    for folder, audit_path in ((reference, Path('output/fsm-linear-controls-audit-v1/audit.json')),
                               (bla, Path('output/fsm-author-bla-audit-v1/audit.json'))):
        expected = json.loads(audit_path.read_text())['inputs']['files']
        members = {str(p.relative_to(folder)): p for p in folder.rglob('*') if p.is_file()}
        if set(members) != set(expected):
            raise ValueError('extra/missing unaudited parent files: '+str(folder))
        for name, path in members.items():
            if path.is_symlink() or {'sha256': sha(path), 'bytes': path.stat().st_size} != expected[name]:
                raise ValueError('changed audited parent file: '+str(path))
    plot_dir = Path('output/fsm-author-nllfr-plots-v1')
    plot = json.loads((plot_dir/'receipt.json').read_text())
    if plot['inputs'] != pins or plot['scientific_status'] != scientific:
        raise ValueError('plot provenance differs from closed audit')
    expected_sources = {str(Path(p).resolve()) for p in
        ('scripts/plot_fsm_author_nllfr.py', 'scripts/plot_fsm_author_bla.py', 'scripts/plot_fsm_linear_controls.py')}
    expected_outputs = {'benchmark.png', 'benchmark.pdf', 'plotted-values.json'}
    if set(plot['sources']) != expected_sources or set(plot['outputs']) != expected_outputs:
        raise ValueError('plot source/output roster mismatch')
    if any(item['path'] != name for name, item in plot['sources'].items()):
        raise ValueError('plot source path mismatch')
    if any(item['path'] != str((plot_dir/name).resolve()) for name, item in plot['outputs'].items()):
        raise ValueError('plot output path mismatch')
    for group in ('inputs', 'sources', 'outputs'):
        for item in plot[group].values():
            if descriptor(item['path']) != item:
                raise ValueError('changed plot input/source/output: '+item['path'])

    vendor = engineering/'vendor'
    vendor_source = vendor/'freq-statespace'
    entries = [study, evaluation, audit.parent, engineering, plot_dir, reference, bla,
        Path('output/fsm-author-nllfr-plot-engineering-v1'),
        Path('pyproject.toml'), Path('uv.lock'),
        vendor_source/'src', vendor_source/'LICENSE', vendor_source/'README.md', vendor_source/'pyproject.toml',
        Path('output/fsm-linear-controls-audit-v1'), Path('output/fsm-author-bla-audit-v1'),
        Path('output/fsm-author-audit-engineering-v1'),
        Path('output/fsm-linear-controls-engineering-v1/original-process.json'),
        Path('output/fsm-correction-study-v1/normalizer.npz'),
        Path('research/fsm-linear-controls-registration.json'), Path('research/fsm_author'),
        Path('research/fsm-author-nllfr-results'), Path('research/fsm-author-bla-results'),
        Path('research/fsm-linear-controls-results/DATA_LICENSE.txt'),
        Path('src/openjev/research/fsm_data.py'), Path('scripts/audit_fsm_author_nllfr.py'),
        Path('scripts/plot_fsm_author_nllfr.py'), Path('scripts/plot_fsm_author_bla.py'),
        Path('scripts/plot_fsm_linear_controls.py'), Path(__file__).relative_to(root)]
    entries.extend(sorted(Path('research').glob('fsm-author-*.*')))
    entries.extend(sorted(Path('research').glob('fsm-nllfr-*.md')))
    entries.extend(sorted(Path('tests').glob('*fsm_author*')))
    files = set()
    excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache'}
    for entry in entries:
        if entry.is_symlink():
            raise ValueError('symlink evidence root: '+str(entry))
        if not entry.exists():
            raise FileNotFoundError(entry)
        if entry.is_file():
            files.add(entry)
        else:
            for directory, dirs, names in os.walk(entry):
                # Never enumerate/read the bundled upstream examples or data.
                # Only the explicit vendor source/license entries above are admitted.
                dirs[:] = sorted(d for d in dirs if d not in excluded and Path(directory)/d != vendor)
                if any((Path(directory)/name).is_symlink() for name in dirs):
                    raise ValueError('symlink directory in evidence')
                files.update(Path(directory)/name for name in names if not name.endswith('.pyc'))
    for path in files:
        if path.is_symlink() or path.name == 'combined_data.npz' or '..' in path.parts:
            raise ValueError('forbidden archive member: '+str(path))
        if path.is_relative_to(vendor) and not (path.is_relative_to(vendor_source/'src')
                or path in {vendor_source/name for name in ('LICENSE', 'README.md', 'pyproject.toml')}):
            raise ValueError('vendor member outside source/license allowlist: '+str(path))
        if path.suffix == '.npy' and ('300mV' in path.name or '_test' in path.name):
            raise ValueError('reserved measurement filename: '+str(path))
    payload = {str(path): {'sha256': sha(path), 'bytes': path.stat().st_size}
               for path in sorted(files)}
    registration = json.loads(Path('research/fsm-author-nllfr-registration.json').read_text())
    if any(item['sha256'] == registration['data_sha256'] for item in payload.values()):
        raise ValueError('original measurement archive excluded from publication')
    # Authenticate again after enumeration, before writing an archive.
    if authenticate(*args)[1] != pins:
        raise ValueError('audited inputs changed while enumerating evidence')
    destination = Path('output/fsm-author-nllfr-publication-v1')
    destination.mkdir(exist_ok=False)
    manifest = {'scientific_status': scientific,
        'scope': 'Closed NL-LFR fit and diagnostic evaluation, audit, plots and fixed comparison evidence. '
                 'Incomplete remains incomplete. Original measurement archive and runtimes excluded; '
                 'FIT-derived arrays and exposed-DEV contexts/predictions included under CC BY 4.0.',
        'files': payload, 'payload_files': len(payload),
        'payload_bytes': sum(item['bytes'] for item in payload.values())}
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    (destination/'evidence-manifest.json').write_bytes(manifest_bytes)
    archive = destination/'evidence.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:  # noqa: SIM117
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in sorted(files):
                data = path.read_bytes()
                if len(data) != payload[str(path)]['bytes'] or hashlib.sha256(data).hexdigest() != payload[str(path)]['sha256']:
                    raise ValueError('file changed during packaging: '+str(path))
                info = tarfile.TarInfo(str(path))
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo('evidence-manifest.json')
            info.size, info.mtime, info.mode = len(manifest_bytes), 0, 0o644
            tar.addfile(info, io.BytesIO(manifest_bytes))
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        if len(names) != len(set(names)) or set(names) != {*payload, 'evidence-manifest.json'}:
            raise ValueError('archive roster mismatch')
        if tar.extractfile('evidence-manifest.json').read() != manifest_bytes:
            raise ValueError('archive manifest mismatch')
        for name, expected in payload.items():
            data = tar.extractfile(name).read()
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('archive byte mismatch: '+name)
    if authenticate(*args)[1] != pins:
        raise ValueError('audited inputs changed during packaging')
    receipt = {'status': 'PASS', 'scientific_status': scientific, 'archive': str(archive),
        'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
        'manifest_sha256': sha(destination/'evidence-manifest.json'),
        'payload_files': len(payload), 'script_sha256': sha(__file__),
        'data_attribution': 'FSM, Merijn Floren, KU Leuven, Floren et al. ISMA-USD 2024; CC BY 4.0',
        'author_code_license': 'GPL-3.0-or-later; source and license included'}
    write(destination/'evidence-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
