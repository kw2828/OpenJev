"""Publish derived robot evidence, retaining source pins but not measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def pin(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.study / 'manifest.json').read_text())['files']
    receipt = json.loads((args.study / 'receipt.json').read_text())
    audit = json.loads(args.audit.read_text())
    if receipt['status'] != 'PASS' or audit['status'] != 'PASS':
        raise ValueError('Closed original campaign and passing evidence audit required')
    for key, name in [('producer_receipt', 'receipt.json'), ('manifest', 'manifest.json')]:
        actual = pin(args.study / name)
        if any(audit['inputs'][key][field] != actual[field] for field in ('bytes', 'sha256')):
            raise ValueError('Audit and original source disagree: ' + name)
    for name, expected in original.items():
        if pin(args.study / name) != expected:
            raise ValueError('Original payload changed: ' + name)
    excluded = [name for name in original if name.startswith(('fit-data-', 'dev-data-', 'dev-windows-'))]
    if len(excluded) != 11:
        raise ValueError('Expected nine measurement arrays and two target-window arrays')
    args.output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in [*original, 'manifest.json', 'receipt.json']:
        if name in excluded:
            continue
        destination = args.output / 'study' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.study / name, destination)
        if pin(destination) != pin(args.study / name):
            raise ValueError('Copy changed bytes: ' + name)
        copied.append('study/' + name)
    shutil.copyfile(args.audit, args.output / 'audit.json')
    repair = args.output / 'audit-repair'
    repair.mkdir()
    for name in ('layout-diagnostic-01.json', 'layout-repair-01.json',
                 'layout-regression-01.py', 'layout-regression-01.log'):
        shutil.copyfile(args.audit.parent / name, repair / name)
    for name in ('benchmark.png', 'benchmark.pdf', 'table.md', 'all-candidates.csv',
                 'plotted-values.json', 'receipt.json'):
        destination = 'plot-receipt.json' if name == 'receipt.json' else name
        shutil.copyfile(args.plot / name, args.output / destination)
    engineering = Path('output/robot-coupling-engineering-v1')
    for name in ('qualification-01.json', 'qualification-01-0.txt', 'qualification-01-1.txt',
                 'environment-01.json', 'root-timing-01.json', 'run-process-01.json', 'run-process-01.log',
                 'audit-process-01.json', 'audit-process-01.log', 'audit-process-02.json', 'audit-process-02.log',
                 'audit_robot_coupling-pre-layout-fix-01.py', 'plot-process-01.json', 'plot-process-01.log'):
        folder = args.output / 'engineering'
        folder.mkdir(exist_ok=True)
        shutil.copyfile(engineering / name, folder / name)
    metadata = Path('output/robot-data-engineering-v1')
    for name in ('contract-metadata.json', 'official-metadata.xml', 'raw-extraction-01.json', 'retrieval.json'):
        folder = args.output / 'source-metadata'
        folder.mkdir(exist_ok=True)
        shutil.copyfile(metadata / name, folder / name)
    (args.output / 'README.md').write_text(
        '# Measured robot coupling evidence\n\n'
        'Derived predictions, all attempted model checkpoints, optimizer states, training traces, '
        'batch schedules, numerical checks and charts from the original registered run. '
        'Failed attempts remain included. The source manifest preserves the hashes of all original files.\n\n'
        'The nine decoded measurement files and two measurement-target window files remain local. '
        'They are not distributed here. Obtain the original '
        '[Industrial Robot data](https://doi.org/10.26204/data/5) and use the pinned causal loader '
        'to reproduce them. The source metadata declares MIT and CC BY-SA 4.0 without a clear '
        'per-file assignment; the repository license is not a relicensing of that data.\n\n'
        'This internal conditional-prediction protocol differs from the published benchmark. '
        'Future inputs are realized measured torques. Confirmation recordings and official TEST '
        'were not decoded. See the [report](../robot-coupling-results.md) and '
        '[frozen protocol](../robot-coupling-protocol.md).\n'
    )
    manifest = {
        'scope': 'Derived evidence only; raw measurements and targets excluded, original pins retained',
        'original_receipt': pin(args.study / 'receipt.json'),
        'original_manifest': pin(args.study / 'manifest.json'),
        'excluded_measurement_payloads': {name: original[name] for name in excluded},
        'copied_original_files': len(copied),
        'files': {str(p.relative_to(args.output)): pin(p)
                  for p in sorted(args.output.rglob('*')) if p.is_file()},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'files': len(manifest['files']) + 1, 'excluded': len(excluded),
                      'bytes': sum(item['bytes'] for item in manifest['files'].values())}))


if __name__ == '__main__':
    main()
