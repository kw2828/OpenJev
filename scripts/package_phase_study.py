"""Publish verified derived Silverbox evidence without copying measurements."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pin(path):
    value = Path(path).read_bytes()
    return {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'plots', 'engineering', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    plot = read(args.plots / 'plot-receipt.json')
    renderer = ROOT / 'scripts/plot_phase_study.py'
    require(set(plot['outputs']) == {'benchmark.png', 'benchmark.pdf', 'report.md', 'plotted-values.json'},
            'exact presentation output roster')
    require(plot['renderer'] == pin(renderer), 'original renderer source')
    require(plot['source_pins_verified_before_and_after'] is True
            and plot['result_reads_after_original_closure'] is True, 'closed original presentation')
    require(all(plot[k] == 0 for k in ('scientific_array_decodes', 'model_calls',
                                       'optimizer_calls', 'scientific_replays')), 'metadata-only presentation')
    for path, expected in plot['inputs'].items():
        require(pin(path) == expected, 'unchanged presentation input: ' + path)
    for name, expected in plot['outputs'].items():
        require(pin(args.plots / name) == expected, 'unchanged presentation output: ' + name)
    spec = importlib.util.spec_from_file_location('phase_publication_admission', renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(plot['version'] == module.VERSION, 'original presentation version')
    inputs, data = module.authenticate(args.study, args.audit, args.engineering / 'run.json',
                                       args.engineering / 'audit.json')
    require(inputs == plot['inputs'] and data['admission'] == plot['admission'],
            'presentation and independent audit input joins')
    plan = read(args.study / 'registration.json')
    require(pin(args.engineering / 'qualification.json') == plan['qualification'],
            'same original qualification process')
    for name, cap in (('qualification', 300), ('run', 2400), ('audit', 600), ('plot', 300)):
        receipt = read(args.engineering / (name + '.json'))
        require(receipt['state'] == 'EXITED' and receipt['returncode'] == 0
                and receipt['timeout_seconds'] == cap, 'closed original process: ' + name)
        require(receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
                'registered source closure: ' + name)
        require(receipt['wrapper'] == pin(args.engineering / 'invoke.py'), 'same process wrapper')
        require(receipt['log_path'] == name + '.log'
                and pin(args.engineering / receipt['log_path']) == receipt['log'], 'original log: ' + name)
        require(0 < receipt['elapsed_seconds'] <= cap, 'process within frozen cap: ' + name)
    expected_plot = ['.venv/bin/python', 'scripts/plot_phase_study.py',
        '--study', str(args.study), '--audit', str(args.audit),
        '--run-receipt', str(args.engineering / 'run.json'),
        '--audit-receipt', str(args.engineering / 'audit.json'), '--output', str(args.plots)]
    require(receipt['argv'] == expected_plot, 'exact original presentation command')
    manifest = read(args.study / 'manifest.json')
    omitted = {f'data-{name}.npz' for name in ('fit', 'dev_a', 'dev_b')}
    require(omitted <= set(manifest['files']), 'all original measured partitions recorded')
    mapping = {'study/' + name: args.study / name for name in manifest['files'] if name not in omitted}
    mapping['study/manifest.json'] = args.study / 'manifest.json'
    mapping.update({'audit.json': args.audit, 'audit-manifest.json': args.audit.parent / 'manifest.json'})
    for name in ('benchmark.png', 'benchmark.pdf', 'report.md', 'plotted-values.json', 'plot-receipt.json'):
        mapping[name] = args.plots / name
    for name in ('qualification', 'run', 'audit', 'plot'):
        mapping['processes/' + name + '.json'] = args.engineering / (name + '.json')
        mapping['processes/' + name + '.log'] = args.engineering / (name + '.log')
    mapping['processes/invoke.py'] = args.engineering / 'invoke.py'
    before = {name: pin(path) for name, path in mapping.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, path in mapping.items():
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(pin(destination) == before[name], 'exact public copy: ' + name)
    source = {
        'dataset': 'Silverbox, SNLS80mV',
        'page': 'https://www.nonlinearbenchmark.org/benchmarks/silverbox',
        'archive': 'https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view',
        'archive_sha256': '2398dda503ac4c30f46c7a1ba09d8a8864d001cc71bbeb59f207a054cc592d8c',
        'csv': plan['csv'],
        'omitted_original_study_files': {name: manifest['files'][name] for name in sorted(omitted)},
        'raw_csv_and_archive_published': False,
        'reason': 'Source does not establish explicit data redistribution permission; loader software license is separate.',
        'numeric_access': 'FIT, DEV A and DEV B inside official TRAIN only; official TEST unparsed.',
        'public_scope': 'Derived predictions, all model checkpoints and training evidence, source snapshots and receipts.',
        'independent_audit_scope': 'Complete local original arrays, including the three measured partitions omitted here.',
        'reproduction': 'Fetch the pinned official source separately; the public manifest preserves its local evidence hashes.',
        'registration_commit': data['admission']['registration_commit'],
    }
    (args.output / 'source-provenance.json').write_text(json.dumps(source, indent=2, sort_keys=True) + '\n')
    after, after_data = module.authenticate(args.study, args.audit, args.engineering / 'run.json',
                                            args.engineering / 'audit.json')
    require(after == inputs and after_data == data, 'original evidence unchanged during publication')
    require(all(pin(path) == before[name] for name, path in mapping.items()), 'copied sources unchanged')
    publication = {'files': {p.relative_to(args.output).as_posix(): pin(p)
                            for p in sorted(args.output.rglob('*')) if p.is_file()},
        'packager': pin(Path(__file__)), 'study_manifest': pin(args.study / 'manifest.json'),
        'audit': pin(args.audit), 'plot_receipt': pin(args.plots / 'plot-receipt.json'),
        'scientific_array_decodes': 0, 'scientific_fits_or_replays': 0,
        'scope': 'Derived public evidence; three measured-partition arrays and raw measurements remain local.'}
    (args.output / 'manifest.json').write_text(json.dumps(publication, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'public_files': len(publication['files']) + 1,
                      'outcome': data['result']['outcome'], 'scientific_fits_or_replays': 0}))


if __name__ == '__main__':
    main()
