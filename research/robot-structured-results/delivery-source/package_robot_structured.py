"""Copy closed audited structured evidence, excluding only DEV target windows."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import plot_robot_structured as plotter

ENGINEERING = ('qualification-01.json', 'qualification-01-0.log',
               'qualification-02.json', 'qualification-02-0.log', 'qualification-02-1.log',
               'delivery-qualification-01.json', 'delivery-qualification-01-0.log',
               'delivery-qualification-01-1.log',
               'run-launch-01.json', 'run-process-01.json', 'run-process-01.log',
               'audit-process-01.json', 'audit-process-01.log', 'plot-process-01.json', 'plot-process-01.log')
DIAGNOSTIC = ('analyze.py', 'definition.json', 'diagnostic.json', 'receipt.json',
              'analysis-01.log', 'process-01.json', 'interpretation.md', 'launch-01.json')
PLOT_FILES = ('benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'table.md', 'plotted-values.json')


def package(study, audit_path, plot, engineering, diagnostic, output):
    study, audit_path, plot, engineering, diagnostic, output = [Path(p).resolve() for p in (study, audit_path, plot, engineering, diagnostic, output)]
    auth = plotter.authenticate(study, audit_path, engineering)
    plan = auth['plan']
    script_pin = plotter.pin(__file__)
    original = plotter.read_json(study / 'manifest.json')['files']
    plot_receipt = plotter.read_json(plot / 'receipt.json')
    plotter.require(plot_receipt['version'] == plotter.VERSION and plot_receipt['study'] == str(study)
                    and plot_receipt['registration_sha256'] == auth['receipt']['registration_sha256']
                    and plot_receipt['inputs'] == auth['inputs'], 'plot must join the exact audited input roster')
    plotter.require(set(plot_receipt['outputs']) == set(PLOT_FILES), 'exact plot output roster')
    for name in PLOT_FILES:
        plotter.require(plotter.pin(plot / name) == plot_receipt['outputs'][name], 'changed plot: ' + name)
    plot_source = Path(plotter.__file__).resolve()
    pp = plotter.read_json(engineering / 'plot-process-01.json')
    plotter.require(pp['returncode'] == 0 and pp['plot_receipt'] == plotter.pin(plot / 'receipt.json')
                    and pp['script'] == plot_receipt['script'] == plotter.pin(plot_source)
                    and pp['log'] == plotter.pin(engineering / 'plot-process-01.log'), 'original plot process join')
    qualification = plan['qualification']
    plotter.require(Path(qualification['path']) == engineering / 'qualification-02.json', 'passing original qualification02 required')
    for number in ('01', '02'):
        q = plotter.read_json(engineering / f'qualification-{number}.json')
        plotter.require(q['sources'] == plan['sources'], 'both qualifications retain the scientific source closure')
        if number == '01':
            plotter.require(q['status'] == 'FAIL' and len(q['commands']) == 1
                            and q['commands'][0]['command'] == ['.venv/bin/ruff', 'check',
                                *(name for name in plotter.SOURCES if name.endswith('.py')),
                                'scripts/launch_robot_structured.py']
                            and q['commands'][0]['returncode'] == 1
                            and Path(q['commands'][0]['log']) == engineering / 'qualification-01-0.log',
                            'original first qualification stopped at lint before tests')
        else:
            plotter.require(q['status'] == 'PASS' and len(q['commands']) == 2
                            and all(row['returncode'] == 0 for row in q['commands']), 'passing second qualification')
        for command in q['commands']:
            plotter.require(plotter.pin(command['log'])['sha256'] == command['sha256'], 'qualification log changed')
    diagnostics = {Path(item['path']).name: item for item in plan['diagnostic_evidence'].values()}
    plotter.require(set(diagnostics) == set(DIAGNOSTIC) and len(diagnostics) == len(plan['diagnostic_evidence']), 'all eight original diagnostic files required')
    for name, item in diagnostics.items():
        plotter.require(Path(item['path']) == diagnostic / name
                        and plotter.pin(diagnostic / name) == {k: item[k] for k in ('sha256', 'bytes')}, 'original diagnostic pin/path')
    excluded = {name: expected for name, expected in original.items() if name.startswith('dev-windows-')}
    expected_excluded = {'dev-windows-' + name + '.npz' for name in plan['config']['partitions']['dev']}
    plotter.require(set(excluded) == expected_excluded and not any(name.startswith(('fit-data-', 'dev-data-')) for name in original), 'exclude exactly two source-target payloads')
    plotter.require(not output.is_relative_to(study) and not output.is_relative_to(plot), 'publication outside immutable inputs')
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    def copy(source, target):
        expected = plotter.pin(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle, Path(source).open('rb') as reader:
            shutil.copyfileobj(reader, handle)
        plotter.require(plotter.pin(target) == expected == plotter.pin(source), 'byte-exact publication copy')
        copied[str(source)] = expected
    for name in [*original, 'receipt.json', 'manifest.json']:
        if name not in excluded:
            copy(study / name, output / 'study' / name)
    copy(audit_path, output / 'audit.json')
    copy(audit_path.parent / 'manifest.json', output / 'audit-manifest.json')
    for name in (*PLOT_FILES, 'receipt.json'):
        copy(plot / name, output / ('plot-receipt.json' if name == 'receipt.json' else name))
    for name in ENGINEERING:
        copy(engineering / name, output / 'engineering' / name)
    for name in DIAGNOSTIC:
        copy(diagnostic / name, output / 'prior-diagnostic' / name)
    for path in (Path(__file__).resolve(), plot_source, plotter.ROOT / 'scripts/audit_robot_structured.py',
                 plotter.ROOT / 'scripts/launch_robot_structured.py',
                 plotter.ROOT / 'tests/test_audit_robot_structured.py',
                 plotter.ROOT / 'tests/test_robot_structured_publication.py'):
        copy(path, output / 'delivery-source' / path.name)
    # Parent closure JSONs are small and remain directly available in the package.
    # Parent measurement arrays stay external under their exact registration pins.
    for name, item in plan['parent_closure'].items():
        copy(Path(item['path']), output / 'parent-closure' / (name + '.json'))
    (output / 'README.md').write_text(
        '# Structured robot transition evidence\n\n'
        'One frozen development campaign: 30 fresh attempts and six copied legacy fits. '
        'Every initial/final checkpoint, Adam state, trace, paired batch, causal ridge '
        'bank, available forecast and scalar result is preserved. Cached parent fit times '
        'are historical; selected-model inference timings were measured on the current host.\n\n'
        'The original manifest retains the hashes of exactly two DEV target-window files '
        'that stay local. The eleven inherited measurement/reference files remain external '
        'under their original pins. Obtain the [Industrial Robot data](https://doi.org/10.26204/data/5) '
        'and use the pinned preprocessing code for source reconstruction. The repository '
        'license does not relicense the original measurements.\n\n'
        'The same DEV2 recordings informed development. CONFIRM2 and official TEST remain '
        'closed. Realized measured torques are conditional forecast inputs, not authenticated '
        'issued actions. This is not independent confirmation, a robot-control result, '
        'a Rust speedup or an architectural novelty claim. Both failed qualification01 and '
        'passing qualification02 are retained; the first stopped at launcher lint before tests. '
        'The prior diagnostic is descriptive intervention evidence from the closed parent.\n\n'
        'See the [report](../robot-structured-results.md) and '
        '[protocol](../robot-structured-protocol.md).\n'
    )
    plotter.require(plotter.authenticate(study, audit_path, engineering) == auth
                    and plotter.pin(__file__) == script_pin, 'evidence/source changed during publication')
    for path, expected in copied.items():
        plotter.require(plotter.pin(path) == expected, 'copy source changed after publication')
    manifest = {'scope': 'Derived evidence only; excludes exactly two future-position target arrays',
                'original_manifest': plotter.pin(study / 'manifest.json'), 'original_receipt': plotter.pin(study / 'receipt.json'),
                'audit': plotter.pin(audit_path), 'packager': script_pin, 'plot_receipt': plotter.pin(plot / 'receipt.json'),
                'excluded_target_payloads': excluded, 'external_measurement_pins': plan['data'],
                'files': {str(p.relative_to(output)): plotter.pin(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return {'files': len(manifest['files']) + 1, 'bytes': sum(p['bytes'] for p in manifest['files'].values()),
            'excluded_targets': len(excluded), 'manifest': plotter.pin(output / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'audit', 'plot', 'engineering', 'diagnostic', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.audit, args.plot, args.engineering, args.diagnostic, args.output)))
