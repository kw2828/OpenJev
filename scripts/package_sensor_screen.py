"""Publish derived sensor-screen evidence without redistributing the source CSV."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pin(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--plots', required=True, type=Path)
    p.add_argument('--engineering', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    for name in ('qualification', 'run', 'audit', 'plot'):
        receipt = read(a.engineering/(name+'.json'))
        require(receipt['state'] == 'EXITED' and receipt['returncode'] == 0,
                'closed original process: '+name)
        require(pin(a.engineering/receipt['log_path']) == receipt['log'], 'original log: '+name)
    study_manifest = read(a.study/'manifest.json')
    for path, expected in study_manifest['files'].items():
        require(pin(a.study/path) == expected, 'study file changed: '+path)
    plan = read(a.study/'registration.json')
    for path, expected in plan['sources'].items():
        require(pin(ROOT/path) == expected == pin(a.study/'sources'/path), 'frozen source changed')
    audit = read(a.audit)
    require(audit['agreement'] is True and audit['admission']['manifest'] == pin(a.study/'manifest.json'),
            'audit agreement and study join')
    require(audit['admission']['run_receipt'] == pin(a.engineering/'run.json'), 'original audit/run join')
    require(read(a.study/'results.json')['outcome'] == audit['results']['outcome'], 'same disposition')
    require(all((a.plots/f).is_file() for f in ('benchmark.png', 'benchmark.pdf', 'report.md',
                                               'plotted-values.json', 'plot-receipt.json')), 'complete plot outputs')
    plotted = read(a.plots/'plot-receipt.json')
    require(plotted['npz_decodes'] == plotted['model_calls'] == plotted['scientific_replays'] == 0,
            'metadata-only delivery')
    require(plotted['source_pins_verified_before_and_after'] is True, 'plot source closure')
    require(plotted['renderer'] == pin(ROOT/'scripts/plot_sensor_screen.py'), 'original plot source')
    for path, expected in plotted['inputs'].items():
        require(pin(path) == expected, 'plot input changed: '+path)
    for name, expected in plotted['outputs'].items():
        require(pin(a.plots/name) == expected, 'plot output changed: '+name)
    require(read(a.audit.parent/'manifest.json') == {'files': {a.audit.name: pin(a.audit)}},
            'audit output manifest')
    mapping = {
        'results.json': a.study/'results.json', 'resources.json': a.study/'resources.json',
        'study-manifest.json': a.study/'manifest.json', 'run-metadata.json': a.study/'run.json',
        'registration.json': a.study/'registration.json', 'audit.json': a.audit,
        'audit-manifest.json': a.audit.parent/'manifest.json', 'report.md': a.plots/'report.md',
        'benchmark.png': a.plots/'benchmark.png', 'benchmark.pdf': a.plots/'benchmark.pdf',
        'plotted-values.json': a.plots/'plotted-values.json', 'plot-receipt.json': a.plots/'plot-receipt.json',
    }
    for name in ('qualification', 'run', 'audit', 'plot'):
        mapping[name+'-process.json'] = a.engineering/(name+'.json')
        mapping[name+'.log'] = a.engineering/(name+'.log')
    a.output.mkdir(parents=True, exist_ok=False)
    for name, path in mapping.items():
        shutil.copyfile(path, a.output/name)
    source = {
        'dataset': 'Air Quality, Saverio Vito (2008), UCI, DOI10.24432/C59K5F',
        'page': 'https://archive.ics.uci.edu/dataset/360/air+quality',
        'download': 'https://archive.ics.uci.edu/static/public/360/air%2Bquality.zip',
        'zip_sha256': 'd4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a',
        'csv': plan['csv'],
        'license_note': 'UCI shows CC BY4.0 and also older research-only/no-commercial language; raw data retained locally.',
        'omitted': ['raw CSV/ZIP', 'train.npz', 'predictions.npz', 'states.npz'],
        'numeric_access': 'March-June2004 only; later measurements unparsed',
        'original_study_manifest_included': True,
        'registration_commit': audit['admission']['registration_commit'],
    }
    (a.output/'source-provenance.json').write_text(json.dumps(source, indent=2, sort_keys=True)+'\n')
    manifest = {'files': {p.name: pin(p) for p in sorted(a.output.iterdir()) if p.is_file()},
                'scope': 'Derived public evidence. Complete original arrays remain in local study; raw source is publicly fetchable.',
                'packager': pin(Path(__file__)), 'audit_input': pin(a.audit),
                'study_manifest': pin(a.study/'manifest.json'), 'plot_input': pin(a.plots/'plot-receipt.json'),
                'scientific_fits_or_replays': 0}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'public_files': len(manifest['files'])+1, 'outcome': audit['results']['outcome'],
                      'scientific_fits_or_replays': 0}))


if __name__ == '__main__':
    main()
