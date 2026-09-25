"""Publish a closed audited transition study without source measurements."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from package_robot_coupling import pin

ENGINEERING = ('qualification-01.json', 'qualification-01-0.log', 'qualification-01-1.log',
               'qualification-02.json', 'qualification-02-0.log', 'qualification-02-1.log',
               'run-process-01.json', 'run-process-01.log', 'audit-process-01.json', 'audit-process-01.log',
               'plot-process-01.json', 'plot-process-01.log')
DIAGNOSTIC = ('analyze.py', 'definition.json', 'diagnostic.json', 'receipt.json', 'analysis-01.log', 'process-01.json')


def regular(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('Regular nonsymlink file required: ' + str(path))
    return pin(path)


def package(study, audit_path, plot, engineering, diagnostic, output):
    study, audit_path, plot, engineering, diagnostic, output = [p.resolve() for p in (study, audit_path, plot, engineering, diagnostic, output)]
    receipt = json.loads((study / 'receipt.json').read_text())
    original = json.loads((study / 'manifest.json').read_text())['files']
    audit = json.loads(audit_path.read_text())
    if receipt['status'] != 'PASS' or audit['status'] != 'PASS' or not audit['agreement']:
        raise ValueError('Closed campaign and passing audit required')
    for key, name in (('manifest', 'manifest.json'), ('producer_receipt', 'receipt.json')):
        if regular(study / name) != {k: audit['inputs'][key][k] for k in ('sha256', 'bytes')}:
            raise ValueError('Audit belongs to a different study payload')
    audit_manifest = json.loads((audit_path.parent / 'manifest.json').read_text())
    if audit_manifest != {'files': {audit_path.name: regular(audit_path)}}:
        raise ValueError('Audit manifest mismatch')
    ap = json.loads((engineering / 'audit-process-01.json').read_text())
    rp = json.loads((engineering / 'run-process-01.json').read_text())
    if (ap['returncode'] != 0 or rp['returncode'] != 0
            or ap['audit_output'] != regular(audit_path)
            or ap['log'] != regular(engineering / 'audit-process-01.log')
            or regular(engineering / 'run-process-01.json') != {k: audit['inputs']['run_receipt'][k] for k in ('sha256', 'bytes')}
            or regular(engineering / 'run-process-01.log') != {k: audit['inputs']['run_log'][k] for k in ('sha256', 'bytes')}
            or rp['registration_sha256'] != receipt['registration_sha256']):
        raise ValueError('Original execution and audit process mismatch')
    plot_receipt = json.loads((plot / 'receipt.json').read_text())
    if (plot_receipt['version'] != 'robot-transition-plot-v1' or plot_receipt['study'] != str(study)
            or plot_receipt['registration_sha256'] != receipt['registration_sha256']
            or plot_receipt['inputs'][str(study / 'manifest.json')] != regular(study / 'manifest.json')):
        raise ValueError('Plot belongs to a different study')
    plot_files = ('benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'table.md', 'plotted-values.json')
    if set(plot_receipt['outputs']) != set(plot_files):
        raise ValueError('Unexpected plot output roster')
    for name in plot_files:
        if regular(plot / name) != plot_receipt['outputs'][name]:
            raise ValueError('Changed plot: ' + name)
    for path, expected in plot_receipt['inputs'].items():
        if regular(Path(path)) != expected:
            raise ValueError('Plot input changed: ' + path)
    pp = json.loads((engineering / 'plot-process-01.json').read_text())
    if (pp['returncode'] != 0 or pp['plot_receipt'] != regular(plot / 'receipt.json')
            or pp['script'] != plot_receipt['script']
            or pp['script'] != regular(Path(__file__).with_name('plot_robot_transition.py'))
            or pp['log'] != regular(engineering / 'plot-process-01.log')):
        raise ValueError('Original plot process mismatch')
    for name, expected in original.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or regular(study / name) != expected:
            raise ValueError('Changed original payload: ' + name)
    plan = json.loads((study / 'registration.json').read_text())
    if regular(engineering / 'qualification-02.json') != {k: plan['qualification'][k] for k in ('sha256', 'bytes')}:
        raise ValueError('Frozen qualification mismatch')
    for number in ('01', '02'):
        qualification = json.loads((engineering / ('qualification-' + number + '.json')).read_text())
        for command in qualification['commands']:
            if regular(Path(command['log']))['sha256'] != command['sha256']:
                raise ValueError('Qualification log changed')
    dr = json.loads((diagnostic / 'receipt.json').read_text())
    if (regular(diagnostic / 'diagnostic.json') != {k: plan['diagnostic'][k] for k in ('sha256', 'bytes')}
            or dr['status'] != 'PASS' or dr['output'] != regular(diagnostic / 'diagnostic.json')
            or dr['script'] != regular(diagnostic / 'analyze.py')):
        raise ValueError('Original descriptive diagnostic mismatch')
    for folder, names in ((engineering, ENGINEERING), (diagnostic, DIAGNOSTIC)):
        for name in names:
            regular(folder / name)
    excluded = {name: expected for name, expected in original.items() if name.startswith('dev-windows-')}
    if len(excluded) != 2 or any(name.startswith(('fit-data-', 'dev-data-')) for name in original):
        raise ValueError('Unexpected measurement payload roster')
    output.mkdir(parents=True, exist_ok=False)
    def copy(source, target):
        regular(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if pin(source) != pin(target):
            raise ValueError('Copy changed bytes')
    for name in [*original, 'receipt.json', 'manifest.json']:
        if name not in excluded:
            copy(study / name, output / 'study' / name)
    copy(audit_path, output / 'audit.json')
    copy(audit_path.parent / 'manifest.json', output / 'audit-manifest.json')
    for name in (*plot_files, 'receipt.json'):
        copy(plot / name, output / ('plot-receipt.json' if name == 'receipt.json' else name))
    for name in ENGINEERING:
        copy(engineering / name, output / 'engineering' / name)
    for name in DIAGNOSTIC:
        copy(diagnostic / name, output / 'prior-diagnostic' / name)
    (output / 'README.md').write_text(
        '# Bounded robot transition evidence\n\n'
        'All attempted checkpoints, optimizer states, traces, sampled batches, causal ridge '
        'coefficients, predictions and scalar results from one frozen development campaign. '
        'The original manifest retains target hashes; two target-window arrays stay local. '
        'The eleven parent measurement/reference files are linked by their original pins, '
        'not redistributed here. Obtain the [Industrial Robot data](https://doi.org/10.26204/data/5) '
        'to reconstruct source measurements with the pinned parent preprocessing code.\n\n'
        'The same DEV2 recordings were previously exposed. This is adaptive development '
        'evidence, not official benchmark performance. Inputs are realized measured torques, '
        'not verified issued commands. CONFIRM2 and official TEST remain closed. '
        'The prior-diagnostic folder is descriptive analysis of the preceding study, '
        'not an additional independent evaluation.\n\n'
        'See the [report](../robot-transition-results.md) and '
        '[frozen protocol](../robot-transition-protocol.md). The repository license '
        'does not relicense the original robot measurements.\n'
    )
    manifest = {'scope': 'Derived evidence only; no raw measurements or future-position targets',
                'original_manifest': pin(study / 'manifest.json'), 'original_receipt': pin(study / 'receipt.json'),
                'excluded_target_payloads': excluded,
                'files': {str(p.relative_to(output)): pin(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {'files': len(manifest['files']) + 1, 'bytes': sum(item['bytes'] for item in manifest['files'].values()), 'excluded_targets': len(excluded)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'audit', 'plot', 'engineering', 'diagnostic', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.audit, args.plot, args.engineering, args.diagnostic, args.output)))
