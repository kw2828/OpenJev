"""Byte-exact publication of closed reflection-capacity evidence, without replay.

Only the two saved DEV target-window payloads are excluded from the child study.
Inherited measurement arrays remain external. Forecasts and checkpoints are
copied as opaque bytes; no numerical decoder, model or fitter is imported.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import plot_robot_reflection_capacity as plotter

VERSION = 'robot-reflection-capacity-package-v1'
PLOT_FILES = ('benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'table.md', 'plotted-values.json')
FIT_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')
DELIVERY_FILES = ('scripts/plot_robot_reflection_capacity.py', 'scripts/package_robot_reflection_capacity.py',
                  'tests/test_robot_reflection_capacity_publication.py',
                  'scripts/audit_robot_reflection_capacity.py', 'tests/test_audit_robot_reflection_capacity.py',
                  'scripts/launch_robot_reflection_capacity.py')
PUBLICATION_FIXTURE_SOURCES = DELIVERY_FILES[:3]


def study_roster(original, plan):
    """Allow only authored derived payloads; future-position targets stay local."""
    cfg = plan['config']
    keys = {f'{arm}-{seed}-lr{ri}' for arm in plotter.ARMS for seed in cfg['seeds'] for ri in range(2)}
    required = {'registration.json', 'runtime.json', 'normalizers.npz', 'parent-normalizers.npz',
                'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json', 'causal_ridge_100.npz',
                'causal_ridge_100.json', 'parent-fits.json', 'checkpoint-barrier.json',
                'results.json', 'resources.json', 'fits.json'}
    required |= {f'batches-{seed}.npz' for seed in cfg['seeds']}
    required |= {f'completed-fit-{i:02d}.json' for i in range(1, 43)}
    required |= {f'{key}/{name}' for key in keys for name in FIT_FILES}
    required |= {'sources/' + name for name in plotter.SOURCES}
    targets = {'dev-windows-' + name + '.npz' for name in cfg['partitions']['dev']}
    references = {'prediction-' + name + '-' + arm + '.npz' for name in cfg['partitions']['dev'] for arm in plotter.REFERENCES}
    predictions = {'prediction-' + name + '-' + key + '.npz' for name in cfg['partitions']['dev'] for key in keys}
    required |= targets | references
    plotter.require(required <= set(original) <= required | predictions, 'exact authored child file types; finite-fit forecasts optional')
    return {name: original[name] for name in sorted(targets)}


def _absolute(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (plotter.ROOT / value).resolve()


def plot_admission(plot, engineering, auth):
    receipt = plotter.read_json(plot / 'receipt.json')
    plotter.require(receipt['version'] == plotter.VERSION and receipt['study'] == auth['study']
                    and receipt['registration_sha256'] == plotter.PLAN_SHA
                    and receipt['inputs'] == auth['inputs'], 'plot joins exact authenticated evidence')
    plotter.require(set(receipt['outputs']) == set(PLOT_FILES), 'exact plot output roster')
    plotter.require({p.name for p in plot.iterdir() if p.is_file()} == {*PLOT_FILES, 'receipt.json'}
                    and not any(p.is_symlink() or p.is_dir() for p in plot.iterdir()), 'exclusive plot file roster')
    for name in PLOT_FILES:
        plotter.require(plotter.pin(plot / name) == receipt['outputs'][name], 'plot bytes changed: ' + name)
    source = Path(plotter.__file__).resolve()
    process = plotter.read_json(engineering / 'plot-process-01.json')
    plotter.require(process['returncode'] == 0 and process['plot_receipt'] == plotter.pin(plot / 'receipt.json')
                    and process['script'] == receipt['script'] == plotter.pin(source)
                    and process['log'] == plotter.pin(engineering / 'plot-process-01.log'), 'original plot process join')
    argv = process['command']
    plotter.require(len(argv) == 10 and argv[0] == '.venv/bin/python'
                    and argv[2::2] == ['--study', '--output', '--audit', '--engineering'], 'original plot argv')
    plotter.require(_absolute(argv[1]) == source
                    and [_absolute(argv[i]) for i in (3, 5, 7, 9)] == [Path(auth['study']), plot, Path(auth['audit_path']), engineering], 'original plot paths')
    return receipt


def qualification_files(engineering, plan):
    """Preserve actual original component, integrated and delivery logs."""
    paths = set()
    qpath = Path(plan['qualification']['path'])
    plotter.require(qpath == engineering / 'qualification-01.json'
                    and plotter.pin(qpath) == {k: plan['qualification'][k] for k in ('sha256', 'bytes')}, 'original integrated qualification')
    for path in (qpath, engineering / 'delivery-qualification-01.json'):
        q = plotter.read_json(path)
        plotter.require(q['status'] == 'PASS' and bool(q['commands']), 'successful original qualification')
        if path == qpath:
            plotter.require(q['sources'] == plan['sources'], 'integrated source closure')
        else:
            plotter.require(set(q['sources']) == set(DELIVERY_FILES), 'exact delivery source roster')
            for name, expected in q['sources'].items():
                plotter.require(plotter.pin(plotter.ROOT / name) == expected, 'qualified delivery source changed')
        paths.add(path)
        for row in q['commands']:
            log = Path(row['log'])
            plotter.require(log.parent == engineering and row['returncode'] == 0
                            and plotter.pin(log)['sha256'] == row['sha256'], 'original qualification log')
            paths.add(log)
    component = plotter.ROOT / 'output/reflection-capacity-engineering-v1'
    core_path, core_log = component / 'tests-01.json', component / 'tests-01.log'
    core = plotter.read_json(core_path)
    plotter.require(core['returncode'] == 0 and core['sources_before'] == core['sources_after']
                    and core['log'] == plotter.pin(core_log), 'original core qualification')
    for name, expected in core['sources_before'].items():
        plotter.require(plan['sources'][name] == expected, 'qualified core source')
    paths.update((core_path, core_log))
    runner = engineering / 'runner-tests-01'
    preflight = runner / 'preflight.json'
    receipt = runner / 'receipt.json'
    runq = plotter.read_json(receipt)
    plotter.require(runq['status'] == 'PASS' and runq['sources_unchanged'] is True, 'original runner qualification')
    plotter.require({p.name for p in runner.iterdir()} == {'preflight.json', 'receipt.json', '0.log', '1.log'}, 'runner qualification exact four files')
    pre = plotter.read_json(preflight)
    plotter.require(pre['sources'] == runq['sources'] and pre['commands'] == [r['argv'] for r in runq['commands']], 'original runner preflight join')
    paths.update((preflight, receipt))
    for index, row in enumerate(runq['commands']):
        log = Path(row['log'])
        plotter.require(log == runner / f'{index}.log' and row['returncode'] == 0
                        and plotter.pin(log) == row['log_pin'], 'original runner log')
        paths.add(log)
    plotter.require(len(runq['commands']) == 2, 'both runner qualification commands')
    return paths


def publication_fixture_files():
    """Retain the failed caption assertion and corrected fabricated test run."""
    base = plotter.ROOT / 'output/robot-reflection-capacity-publication-engineering-v1'
    paths = set()
    for name, code in (('fixture-01', 1), ('fixture-02', 0), ('fixture-03', 0)):
        folder = base / name
        pre = plotter.read_json(folder / 'preflight.json')
        receipt = plotter.read_json(folder / 'receipt.json')
        plotter.require(receipt['returncode'] == code and pre['sources'] == receipt['sources_before'] == receipt['sources_after']
                        and set(pre['sources']) == set(PUBLICATION_FIXTURE_SOURCES), 'original publication fixture source/outcome')
        plotter.require(pre['command'] == receipt['command'] == ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                        'tests/test_robot_reflection_capacity_publication.py'], 'original publication fixture command')
        plotter.require(plotter.pin(folder / 'tests.log') == receipt['log'], 'original publication fixture log')
        paths.update(folder / f for f in ('preflight.json', 'receipt.json', 'tests.log'))
        for source, expected in pre['sources'].items():
            path = folder / 'sources' / source
            plotter.require(plotter.pin(path) == expected, 'original publication fixture source snapshot')
            paths.add(path)
    return base, paths


def package(study, audit_path, plot, engineering, output):
    study, audit_path, plot, engineering, output = [Path(p).resolve() for p in (study, audit_path, plot, engineering, output)]
    auth = plotter.authenticate(study, audit_path, engineering)
    plan = auth['plan']
    script_pin = plotter.pin(__file__)
    original = plotter.read_json(study / 'manifest.json')['files']
    excluded = study_roster(original, plan)
    plot_receipt = plot_admission(plot, engineering, auth)
    engineering_paths = qualification_files(engineering, plan)
    fixture_root, fixture_paths = publication_fixture_files()
    engineering_paths |= {engineering / name for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log',
                            'audit-process-01.json', 'audit-process-01.log', 'plot-process-01.json', 'plot-process-01.log')}
    plotter.require(all(not output.is_relative_to(folder) for folder in (study, plot, engineering, audit_path.parent)), 'publication outside immutable inputs')
    output.mkdir(parents=True, exist_ok=False)
    copied, mapping = {}, {}
    def copy(source, relative):
        source = Path(source).resolve()
        target = plotter.relative_path(output, relative)
        expected = plotter.pin(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle, source.open('rb') as reader:
            shutil.copyfileobj(reader, handle)
        plotter.require(plotter.pin(target) == expected == plotter.pin(source), 'byte-exact evidence copy')
        copied[str(source)] = expected
        mapping[relative] = {'path': str(source), **expected}
    for name in [*original, 'receipt.json', 'manifest.json']:
        if name not in excluded:
            copy(study / name, 'study/' + name)
    copy(audit_path, 'audit.json')
    copy(audit_path.parent / 'manifest.json', 'audit-manifest.json')
    for name in (*PLOT_FILES, 'receipt.json'):
        copy(plot / name, 'plot-receipt.json' if name == 'receipt.json' else name)
    for path in sorted(engineering_paths):
        copy(path, 'engineering/' + str(path.relative_to(engineering)) if path.is_relative_to(engineering)
             else 'core-qualification/' + path.name)
    for path in sorted(fixture_paths):
        copy(path, 'publication-qualification/' + str(path.relative_to(fixture_root)))
    for name in DELIVERY_FILES:
        copy(plotter.ROOT / name, 'delivery-source/' + name)
    for name, item in plan['parent_closure'].items():
        copy(item['path'], 'parent-closure/' + name + '.json')
    copy(plotter.ROOT / 'research/robot-structured-registration.json', 'parent-closure/registration.json')
    # Only audited scalars are used for this short outcome label. No outcomes are recomputed.
    result = auth['audit']['results']['result']
    (output / 'README.md').write_text(
        '# Reflection-capacity evidence\n\n'
        f"Original scientific outcome: **{result['status']}**, {result['passed']}/69 conditions "
        f"(accuracy {result['accuracy']['passed']}/67; compute {result['compute']['passed']}/2). "
        'Successful process closure and an agreeing audit do not convert a failed scientific rule into a pass.\n\n'
        'Six R12 attempts are new; all36 parent recipes, both rates, all three seeds and every failed case remain visible. '
        'Original initial/final checkpoints, Adam arrays, traces, batch indices, reference banks, available forecasts, '
        'scores and all29 frozen source copies are preserved byte-for-byte. Cached fit times are historical; '
        'selected inference measurements were repeated on the current host. The copy map records every original path/hash.\n\n'
        'All three publication-only fixture attempts are retained. The first had one stale caption-text assertion; '
        'the second corrected that assertion, and the third checked common accuracy-panel axes. '
        'These tests do not change the scientific results or criteria.\n\n'
        'Exactly two DEV target-window files are excluded under their original manifest hashes. '
        'The eleven inherited measurement/reference inputs remain external, and no raw MAT or CONFIRM/TEST data is copied. '
        'Reproduction requires those licensed local inputs and the [parent evidence](../robot-structured-results.md). '
        'Original data: [Industrial Robot](https://doi.org/10.26204/data/5). Repository licensing does not relicense measurements.\n\n'
        'This is adaptive development on exposed DEV, with conditional forecasts given realized measured torque. '
        'It is not an independent confirmation, official benchmark score, robot-control result, native speedup or novel architecture claim. '
        'The 806-parameter R12 model has no storage advantage over bounded dense.\n\n'
        'See the [report](../robot-reflection-capacity-results.md) and [protocol](../robot-reflection-capacity-protocol.md).\n'
    )
    plotter.require(plotter.authenticate(study, audit_path, engineering) == auth
                    and plotter.pin(__file__) == script_pin, 'inputs or packager changed')
    plotter.require(plot_admission(plot, engineering, auth) == plot_receipt, 'plot evidence changed')
    for path, expected in copied.items():
        plotter.require(plotter.pin(path) == expected, 'copy source changed after publication')
    manifest = {'version': VERSION, 'scope': 'Opaque derived evidence; exactly two measured DEV target arrays excluded',
                'original_manifest': plotter.pin(study / 'manifest.json'), 'original_receipt': plotter.pin(study / 'receipt.json'),
                'audit': plotter.pin(audit_path), 'packager': script_pin, 'plot_receipt': plotter.pin(plot / 'receipt.json'),
                'excluded_target_payloads': excluded, 'external_measurement_pins': plan['data'], 'copy_sources': mapping,
                'files': {str(p.relative_to(output)): plotter.pin(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return {'files': len(manifest['files']) + 1, 'bytes': sum(p['bytes'] for p in manifest['files'].values()),
            'excluded_targets': len(excluded), 'manifest': plotter.pin(output / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'audit', 'plot', 'engineering', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.audit, args.plot, args.engineering, args.output)))
