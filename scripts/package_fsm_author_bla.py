"""Package the closed BLA comparison without source measurements or runtimes."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

from plot_fsm_author_bla import authenticate, descriptor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    study = Path('output/fsm-author-bla-study-v1')
    audit = Path('output/fsm-author-bla-audit-v1/audit.json')
    audit_process = Path('output/fsm-author-engineering-v1/audit-process.json')
    reference = Path('output/fsm-linear-controls-study-v1')
    _, input_pins = authenticate(study.resolve(), audit.resolve(), audit_process.resolve(), reference.resolve())
    audit_value = json.loads(audit.read_text())
    for directory, key in ((study, 'files'), (reference, 'reference_files')):
        inventory = audit_value['inputs'][key]
        current = {str(p.relative_to(directory)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                   for p in directory.rglob('*') if p.is_file()}
        if current != inventory:
            raise ValueError('audited inventory changed: '+str(directory))
    plot = json.loads(Path('output/fsm-author-bla-plot-v1/receipt.json').read_text())
    if plot['inputs'] != input_pins:
        raise ValueError('plot inputs do not match closed audit')
    for item in (*plot['inputs'].values(), *plot['sources'].values(), *plot['outputs'].values()):
        if descriptor(item['path']) != item:
            raise ValueError('plot provenance changed: '+item['path'])
    if json.loads((study/'summary.json').read_text())['status'] != 'REFERENCE_COMPLETE':
        raise ValueError('complete author reference required')
    destination = Path('output/fsm-author-bla-publication-v1')
    destination.mkdir(exist_ok=False)
    entries = [study, Path('output/fsm-author-bla-audit-v1'), Path('output/fsm-author-audit-engineering-v1'),
        Path('output/fsm-author-engineering-v1'), Path('output/fsm-author-bla-plot-v1'),
        Path('output/fsm-linear-controls-study-v1'), Path('output/fsm-linear-controls-audit-v1/audit.json'),
        Path('output/fsm-linear-controls-engineering-v1/original-process.json'),
        Path('output/fsm-correction-study-v1/normalizer.npz'),
        Path('research/fsm-linear-controls-registration.json'), Path('research/fsm_author'),
        Path('research/fsm-author-bla-results'), Path('src/openjev/research/fsm_data.py'),
        Path('research/fsm-linear-controls-results/DATA_LICENSE.txt'),
        Path('scripts/audit_fsm_author_bla.py'), Path('scripts/plot_fsm_author_bla.py'), Path(__file__).relative_to(root)]
    entries.extend(sorted(Path('research').glob('fsm-author-*.md')))
    entries.extend(Path(item['path']).relative_to(root) for item in plot['sources'].values())
    entries.extend(sorted(Path('tests').glob('*fsm_author*')))
    entries.append(Path('research/fsm-author-bla-registration.json'))
    files = set()
    excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache'}
    for entry in entries:
        if not entry.exists():
            raise FileNotFoundError(entry)
        if entry.is_file():
            files.add(entry)
        else:
            for directory, dirs, names in os.walk(entry):
                dirs[:] = sorted(d for d in dirs if d not in excluded)
                files.update(Path(directory)/name for name in names if not name.endswith('.pyc'))
    for path in files:
        if path.is_symlink() or path.name == 'combined_data.npz':
            raise ValueError('forbidden archive member '+str(path))
    payload = {str(path): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(files)}
    registration = json.loads(Path('research/fsm-author-bla-registration.json').read_text())
    if any(item['sha256'] == registration['data_sha256'] for item in payload.values()):
        raise ValueError('original raw measurement content forbidden')
    manifest = {'scope': 'closed BLA study, engineering/audit/plot and prior comparison evidence; no original measurement archive or virtualenv',
                'files': payload, 'payload_files': len(payload),
                'payload_bytes': sum(item['bytes'] for item in payload.values())}
    manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
    (destination/'evidence-manifest.json').write_bytes(manifest_bytes)
    archive = destination/'evidence.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:  # noqa: SIM117
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in sorted(files):
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != payload[str(path)]['sha256']:
                    raise ValueError('file changed during packaging '+str(path))
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
        for name, expected in payload.items():
            data = tar.extractfile(name).read()
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('archive byte mismatch '+name)
    receipt = {'status': 'PASS', 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
               'archive_sha256': sha(archive), 'manifest_sha256': sha(destination/'evidence-manifest.json'),
               'payload_files': len(payload), 'script_sha256': sha(__file__),
               'data_attribution': 'FSM, Merijn Floren, KU Leuven, Floren et al. ISMA-USD2024; CC BY4.0',
               'author_code_license': 'GPL-3.0-or-later; source and license included'}
    write(destination/'evidence-receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
