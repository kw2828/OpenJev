"""Package closed, audited measurement evidence without numerical replay."""
import hashlib
import importlib.util
import json
import math
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def descriptor(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def main():
    study, delivery, out = (ROOT/p for p in ('output/measurement-v1', 'output/measurement-delivery-v1',
                                           'research/measurement-results'))
    spec = importlib.util.spec_from_file_location('measurement_package_admission', ROOT/'scripts/audit_measurement_study.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    admission = checker.authenticate(study, delivery/'run-process.json')
    audit = json.loads((delivery/'audit.json').read_text())
    plot = json.loads((out/'plot-receipt.json').read_text())
    closure = json.loads((delivery/'plot-process.json').read_text())
    expected = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/plot_measurement_study.py')]
    if (closure['state'] != 'EXITED' or closure['returncode'] != 0
            or not math.isfinite(closure['elapsed_seconds']) or closure['elapsed_seconds'] <= 0
            or [checker.absolute_argument(v) for v in closure['command']] != expected
            or audit['agreement'] is not True or audit['admission'] != admission
            or plot['admission'] != admission or plot['audit'] != descriptor(delivery/'audit.json')
            or plot['audit_process'] != descriptor(delivery/'audit-process.json')
            or plot['renderer'] != descriptor(ROOT/'scripts/plot_measurement_study.py')
            or plot['version'] != 'measurement-plot-v1' or plot['npz_decodes'] != 0 or plot['model_calls'] != 0):
        raise ValueError('closed audited plot provenance required')
    if set(plot['outputs']) != {'benchmark.png', 'benchmark.pdf', 'audit.json', 'summary.json', 'bootstrap.json'}:
        raise ValueError('exact plot output roster')
    for name, pin in plot['outputs'].items():
        if descriptor(out/name) != pin:
            raise ValueError('plot output changed: '+name)
    for name, original in [('audit.json', delivery/'audit.json'), ('summary.json', study/'summary.json'),
                           ('bootstrap.json', study/'bootstrap.json')]:
        if (out/name).read_bytes() != original.read_bytes():
            raise ValueError('published evidence differs: '+name)
    members = {}
    for prefix, directory in [('study', study), ('delivery', delivery),
                              ('engineering/memory', ROOT/'output/measurement-memory-engineering-v1'),
                              ('engineering/audit', ROOT/'output/measurement-audit-engineering-v1')]:
        if not directory.is_dir():
            raise ValueError('required evidence directory: '+str(directory))
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                members[prefix+'/'+str(path.relative_to(directory))] = path
    for name in (*plot['outputs'], 'plot-receipt.json'):
        members['presentation/'+name] = out/name
    for name in ('scripts/plot_measurement_study.py', 'scripts/package_measurement_study.py',
                 'research/measurement-results.md', 'research/measurement-protocol.md',
                 'research/measurement-next.md', 'docs/decision-model-architecture.md',
                 'README.md', 'research/experiment-index.md'):
        members['publication/'+name] = ROOT/name
    parent = ROOT/'research/retention-results/summary.json'
    if descriptor(parent) != admission['parent_result']:
        raise ValueError('parent lineage pin')
    members['parent-lineage/retention-summary.json'] = parent
    if any(path.is_symlink() for path in members.values()):
        raise ValueError('ordinary evidence files only')
    inventory = {name: descriptor(path) for name, path in sorted(members.items())}
    archive = out/'evidence.tar.gz'
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
        for name, path in sorted(members.items()):
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as bundle:
        if set(bundle.getnames()) != set(inventory) or len(bundle.getnames()) != len(inventory):
            raise ValueError('archive member roster')
        for item in bundle.getmembers():
            content = bundle.extractfile(item).read()
            if {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)} != inventory[item.name]:
                raise ValueError('archive byte mismatch')
    if checker.authenticate(study, delivery/'run-process.json') != admission:
        raise ValueError('evidence changed during packaging')
    index = {'version': 'measurement-publication-v1', 'gate': audit['gate'], 'archive': descriptor(archive),
             'member_count': len(inventory), 'members': inventory, 'admission': admission,
             'scope': 'Opaque packaging only. No generation, training, model calls or numerical replay.'}
    with (out/'archive-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'archive': index['archive'], 'member_count': index['member_count']}))


if __name__ == '__main__':
    main()
