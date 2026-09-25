"""Copy closed history evidence as opaque bytes, excluding two measured targets."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import plot_robot_history_initialization as plotter

VERSION = 'robot-history-initialization-package-v1'
SCIENTIFIC_QUALIFICATION_SHA = '29d1b0a7b9683e2cef2f3c35b9d825cbadb3fe450c39c0fabcd8d3dae3f06130'
PLOT_FILES = ('benchmark.png', 'benchmark.pdf', 'all-candidates.csv', 'permutation.csv', 'table.md', 'plotted-values.json')
FIT_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')
PUBLICATION_SOURCES = ('scripts/plot_robot_history_initialization.py', 'scripts/package_robot_history_initialization.py',
                       'tests/test_robot_history_publication.py', 'tests/test_robot_history_package.py')
AUDIT_SOURCES = ('scripts/audit_robot_history_initialization.py', 'tests/test_audit_robot_history_initialization.py')
DELIVERY_FILES = (*PUBLICATION_SOURCES, *AUDIT_SOURCES, 'scripts/launch_robot_history_initialization.py')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def study_roster(original, plan):
    """Whitelisted derived artifacts only; the auditor checks their exact dynamic roster."""
    cfg = plan['config']
    plotter.require(tuple(cfg['comparison_arms']) == plotter.ARMS and len(plan['sources']) == 29
                    and cfg['seeds'] == [8101, 8102, 8103] and cfg['learning_rates'] == [.001, .003], 'registered complete design')
    dev = cfg['partitions']['dev']
    plotter.require(len(dev) == len(set(dev)) == 2, 'two target recordings')
    keys = {f'{arm}-{seed}-lr{ri}' for arm in plotter.ARMS for seed in cfg['seeds'] for ri in range(2)}
    required = {'registration.json', 'runtime.json', 'normalizers.npz', 'linear.npz',
                'causal_ridge_1.npz', 'causal_ridge_1.json', 'causal_ridge_100.npz', 'causal_ridge_100.json',
                'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json', 'permutation.json',
                'parent-structured-fits.json', 'parent-structured-results.json',
                'parent-transition-fits.json', 'parent-transition-results.json'}
    required |= {f'batches-{seed}.npz' for seed in cfg['seeds']}
    required |= {f'completed-fit-{i:02d}.json' for i in range(1, 49)}
    required |= {f'{key}/{name}' for key in keys for name in FIT_FILES}
    required |= {'sources/' + name for name in plan['sources']}
    targets = {f'dev-windows-{name}.npz' for name in dev}
    required |= targets | {f'prediction-{name}-{arm}.npz' for name in dev for arm in plotter.REFERENCES}
    ordinary = {f'prediction-{name}-{key}.npz' for name in dev for key in keys}
    diagnostic = {f'permuted-prediction-{name}-{arm}-{seed}-lr{ri}.npz' for name in dev
                  for arm in plotter.PRIMARY for seed in cfg['seeds'] for ri in range(2)}
    plotter.require(required <= set(original) <= required | ordinary | diagnostic, 'exact derived study file types')
    for name in original:
        plotter.relative_path(Path('/evidence'), name)
    for name in dev:
        for arm in plotter.PRIMARY:
            for seed in cfg['seeds']:
                plotter.require(sum(f'permuted-prediction-{name}-{arm}-{seed}-lr{ri}.npz' in original for ri in range(2)) <= 1,
                                'only the selected diagnostic rate')
    return {name: original[name] for name in sorted(targets)}


study_allowlist = study_roster


def _absolute(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (plotter.ROOT / value).resolve()


def regular_tree(folder, expected):
    plotter.require(folder.is_dir() and not folder.is_symlink(), 'regular evidence directory')
    paths = list(folder.rglob('*'))
    directories = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != '.'}
    plotter.require(not any(p.is_symlink() for p in paths)
                    and {str(p.relative_to(folder)) for p in paths if p.is_file()} == set(expected)
                    and {str(p.relative_to(folder)) for p in paths if p.is_dir()} == directories, 'exact regular evidence tree')
    for name in expected:
        plotter.relative_path(folder, name)


def plot_admission(plot, engineering, auth):
    receipt = plotter.read_json(plot / 'receipt.json')
    plotter.require(receipt['version'] == plotter.VERSION and receipt['study'] == auth['study']
                    and receipt['registration_sha256'] == plotter.PLAN_SHA
                    and receipt['inputs'] == auth['inputs'], 'plot joins authenticated evidence')
    plotter.require(set(receipt['outputs']) == set(PLOT_FILES), 'exact plot outputs')
    regular_tree(plot, {*PLOT_FILES, 'receipt.json'})
    for name in PLOT_FILES:
        plotter.require(plotter.pin(plot / name) == receipt['outputs'][name], 'plot payload changed')
    source = Path(plotter.__file__).resolve()
    process = plotter.read_json(engineering / 'plot-process-01.json')
    plotter.require(process['returncode'] == 0 and process['source_unchanged'] is True
                    and process['plot_receipt'] == plotter.pin(plot / 'receipt.json')
                    and process['script'] == receipt['script'] == plotter.pin(source)
                    and process['log'] == plotter.pin(engineering / 'plot-process-01.log'), 'original plot process joins')
    argv = process['command']
    plotter.require(len(argv) == 10 and argv[0] == '.venv/bin/python'
                    and argv[2::2] == ['--study', '--output', '--audit', '--engineering'], 'original plot argv')
    plotter.require(_absolute(argv[1]) == source and [_absolute(argv[i]) for i in (3, 5, 7, 9)]
                    == [Path(auth['study']), plot, Path(auth['audit_path']), engineering], 'original plot paths')
    return receipt


def qualification_attempt(folder, expected_sources, require_current=False):
    """Preserve an actual stopped or completed attempt without inventing missing logs."""
    pre = plotter.read_json(folder / 'preflight.json')
    receipt = plotter.read_json(folder / 'receipt.json')
    plotter.require(set(pre['sources']) == set(expected_sources) and pre['sources'] == receipt['sources']
                    and receipt['sources_unchanged'] is True
                    and pre['thread_env'] == receipt['thread_env'] == dict.fromkeys(THREADS, '1'), 'qualification source/thread closure')
    rows, commands = receipt['commands'], pre['commands']
    expected_commands = [['.venv/bin/ruff', 'check', *expected_sources],
                         ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                          *[name for name in expected_sources if name.startswith('tests/')]]]
    plotter.require(commands == expected_commands, 'exact qualification Ruff and fabricated-test argv')
    plotter.require(bool(rows) and len(rows) <= len(commands) and receipt['status'] in ('PASS', 'FAILED'), 'original qualification outcome')
    plotter.require([r['command'] for r in rows] == commands[:len(rows)], 'original executed command prefix')
    codes = [r['returncode'] for r in rows]
    plotter.require(all(type(code) is int for code in codes) and all(code == 0 for code in codes[:-1]), 'stop at first qualification failure')
    plotter.require((receipt['status'] == 'PASS' and len(rows) == len(commands) and codes[-1] == 0)
                    or (receipt['status'] == 'FAILED' and codes[-1] != 0), 'qualification status matches executed commands')
    names = {'preflight.json', 'receipt.json'}
    for name, expected in pre['sources'].items():
        snapshot = plotter.relative_path(folder / 'sources', name)
        plotter.require(plotter.pin(snapshot) == expected, 'qualification source snapshot')
        if require_current:
            plotter.require(plotter.pin(plotter.ROOT / name) == expected, 'final qualified source drift')
        names.add('sources/' + name)
    for i, row in enumerate(rows, 1):
        log = folder / f'command-{i}.log'
        plotter.require(Path(row['log']) == log and plotter.pin(log) == row['log_pin']
                        and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] > 0,
                        'original qualification log/duration')
        names.add(log.name)
    regular_tree(folder, names)
    if require_current:
        plotter.require(receipt['status'] == 'PASS', 'final qualification must pass')
    return {folder / name for name in names}


def qualification_files():
    result = {}
    scientific = plotter.ROOT / 'research/robot-history-initialization-qualification-results'
    manifest_path = scientific / 'manifest.json'
    plotter.require(plotter.pin(manifest_path)['sha256'] == SCIENTIFIC_QUALIFICATION_SHA, 'frozen scientific qualification bundle')
    manifest = plotter.read_json(manifest_path)
    plotter.require(len(manifest['files']) == 72, 'all72 scientific qualification payloads')
    regular_tree(scientific, {*manifest['files'], 'manifest.json'})
    for name, expected in manifest['files'].items():
        path = plotter.relative_path(scientific, name)
        plotter.require(plotter.pin(path) == expected, 'scientific qualification bytes')
        result['scientific-qualification/' + name] = path
    result['scientific-qualification/manifest.json'] = manifest_path
    audit_root = plotter.ROOT / 'output/robot-history-initialization-audit-engineering-v1'
    folder = audit_root / 'qualification-01'
    for path in qualification_attempt(folder, AUDIT_SOURCES, require_current=True):
        result['audit-qualification/' + str(path.relative_to(audit_root))] = path
    public_root = plotter.ROOT / 'output/robot-history-publication-engineering-v1'
    attempts = sorted(p for p in public_root.iterdir() if p.is_dir() and p.name.startswith('qualification-'))
    plotter.require(bool(attempts) and [p.name for p in attempts] == [f'qualification-{i:02d}' for i in range(1, len(attempts)+1)],
                    'all consecutive original publication qualifications')
    for folder in attempts:
        for path in qualification_attempt(folder, PUBLICATION_SOURCES, require_current=folder == attempts[-1]):
            result['publication-qualification/' + str(path.relative_to(public_root))] = path
    return result


def package(study, audit_path, plot, engineering, output):
    study, audit_path, plot, engineering, output = [Path(p).resolve() for p in (study, audit_path, plot, engineering, output)]
    auth = plotter.authenticate(study, audit_path, engineering)
    plan, script_pin = auth['plan'], plotter.pin(__file__)
    original = plotter.read_json(study / 'manifest.json')['files']
    excluded = study_roster(original, plan)
    plot_receipt = plot_admission(plot, engineering, auth)
    qualifications = qualification_files()
    protected = (study, plot, engineering, audit_path.parent,
                 plotter.ROOT / 'research/robot-history-initialization-qualification-results',
                 plotter.ROOT / 'output/robot-history-initialization-audit-engineering-v1',
                 plotter.ROOT / 'output/robot-history-publication-engineering-v1',
                 *[p.parent for p in qualifications.values()])
    plotter.require(all(not output.is_relative_to(folder) and not folder.is_relative_to(output) for folder in protected),
                    'publication outside immutable evidence')
    output.mkdir(parents=True, exist_ok=False)
    copied, mapping = {}, {}
    def copy(source, relative):
        source = Path(source)
        expected = plotter.pin(source)
        target = plotter.relative_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle, source.open('rb') as reader:
            shutil.copyfileobj(reader, handle)
        plotter.require(plotter.pin(target) == expected == plotter.pin(source), 'byte-exact evidence copy')
        copied[str(source)] = expected
        mapping[relative] = {'path': str(source.resolve()), **expected}
    for name in [*original, 'receipt.json', 'manifest.json']:
        if name not in excluded:
            copy(study / name, 'study/' + name)
    copy(audit_path, 'audit.json')
    copy(audit_path.parent / 'manifest.json', 'audit-manifest.json')
    for name in (*PLOT_FILES, 'receipt.json'):
        copy(plot / name, 'plot-receipt.json' if name == 'receipt.json' else name)
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log', 'audit-process-01.json',
                 'audit-process-01.log', 'plot-process-01.json', 'plot-process-01.log'):
        copy(engineering / name, 'engineering/' + name)
    for relative, path in sorted(qualifications.items()):
        copy(path, relative)
    for name in DELIVERY_FILES:
        copy(plotter.ROOT / name, 'delivery-source/' + name)
    for group, closure in plan['parent_closure'].items():
        for name, item in closure.items():
            copy(item['path'], f'parent-closure/{group}/{name}.json')
        copy(plotter.ROOT / f'research/robot-{group}-registration.json', f'parent-closure/{group}/registration.json')
    gate = auth['audit']['results']['result']
    plotter.require(gate['total'] == len(gate['conditions']) == 5 and gate['passed'] == sum(c['passed'] for c in gate['conditions'])
                    and tuple(c['name'] for c in gate['conditions']) == plotter.CONDITIONS
                    and gate['status'] == ('QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if gate['passed'] == 5
                                           else 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION'), 'literal five-condition scientific label')
    (output / 'README.md').write_text(
        '# History-initialization evidence\n\n'
        f"Original scientific result: **{gate['status']}**, {gate['passed']}/5 conditions. "
        'An agreeing audit certifies evidence consistency; it does not turn a failed scientific rule into a pass.\n\n'
        'All18 fresh attempts and30 unchanged inherited fits are retained, including both rates and all three seeds. '
        'Initial/final checkpoints, Adam arrays, traces, paired batches, all available ordinary and reversed-history forecasts, '
        '208 ordinary rows,36 diagnostic rows and29 frozen source snapshots are preserved. Known temporal reversal failures '
        'remain separate diagnostic evidence. They are not a sixth scientific gate.\n\n'
        'Cached training times are historical. Selected full-request inference timings were repeated on this host, '
        'including normalization, casting, conditioning, rollout, denormalization and boundary checks. '
        'Storage includes parameters, recurrent state, buffers and normalizers; temporary workspace is not measured. '
        'Each affine initializer adds372 parameters to the590-parameter transition; last-two has590.\n\n'
        'Exactly two measured DEV target-window files are excluded under their original hashes. Eleven inherited input '
        'payloads remain external, with their pins in the manifest; no raw MAT, CONFIRM or official TEST data is copied. '
        'Reproduction requires the licensed local inputs and both '
        '[structured](../robot-structured-results.md) and [transition](../robot-transition-results.md) parent evidence. '
        'Original data: [Industrial Robot](https://doi.org/10.26204/data/5). Repository licensing does not relicense measurements.\n\n'
        'This is an information-control comparison on previously exposed DEV, conditional on realized measured torque. '
        'A pass supports only a separately registered confirmation. It establishes no architecture novelty, control performance, '
        'or native speedup. All original qualification attempts and their exact source snapshots remain visible.\n\n'
        'See the [report](../robot-history-initialization-results.md) and [protocol](../robot-history-initialization-protocol.md).\n'
    )
    plotter.require(plotter.authenticate(study, audit_path, engineering) == auth and plotter.pin(__file__) == script_pin,
                    'inputs or packager changed')
    plotter.require(plot_admission(plot, engineering, auth) == plot_receipt and qualification_files() == qualifications,
                    'presentation or qualification roster changed')
    for path, expected in copied.items():
        plotter.require(plotter.pin(path) == expected, 'original evidence changed during copy')
    for relative, item in mapping.items():
        plotter.require(plotter.pin(output / relative) == {k: item[k] for k in ('sha256', 'bytes')}, 'copied evidence changed')
    manifest = {'version': VERSION, 'scope': 'Opaque derived evidence; exactly two measured DEV targets excluded',
                'original_manifest': plotter.pin(study / 'manifest.json'), 'original_receipt': plotter.pin(study / 'receipt.json'),
                'audit': plotter.pin(audit_path), 'packager': script_pin, 'plot_receipt': plotter.pin(plot / 'receipt.json'),
                'excluded_target_payloads': excluded, 'external_measurement_pins': plan['data'], 'copy_sources': mapping,
                'files': {str(p.relative_to(output)): plotter.pin(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return {'files': len(manifest['files'])+1, 'bytes': sum(p['bytes'] for p in manifest['files'].values()),
            'excluded_targets': len(excluded), 'manifest': plotter.pin(output / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'audit', 'plot', 'engineering', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.study, args.audit, args.plot, args.engineering, args.output)))
