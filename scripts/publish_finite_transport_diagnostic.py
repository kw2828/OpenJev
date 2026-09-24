"""Publish closed saved geometry and an opaque child-evidence archive only.

No checkpoint decoder, model, data generator or numerical diagnostic is called.
The complete parent release remains an explicit dependency; the archive
contains the diagnostic child folder, including both source snapshots.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

import diagnose_finite_transport as diagnostic

ROOT = diagnostic.ROOT
VERSION = 'finite-transport-publication-v1'
REGISTRATION_SHA256 = '3f1aefbd5f869e08c40b5a1f2beefeb1b0c7361a9462860e363a92bcc90c8352'
PARENT_RELEASE = 'https://github.com/kw2828/OpenJev/releases/tag/finite-training-allocation-v1'
FAILED_REGISTRATION_SHA256 = '9714355b0188c18ac1b7ef180adf2f09e3f02c0bf9a73c1266b5087d2caf3a1c'
COUNTS = {'checkpoint_decodes': 27, 'parameter_array_decodes': 108, 'model_calls': 0,
          'optimizer_calls': 0, 'generator_calls': 0, 'new_evaluation_cases': 0,
          'new_training_updates': 0}
PUBLICATION_COUNTS = {'checkpoint_decodes': 0, 'array_decodes': 0, 'model_calls': 0,
                      'optimizer_calls': 0, 'generator_calls': 0, 'diagnostic_recomputations': 0}
require, descriptor = diagnostic.require, diagnostic.descriptor


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'regular absolute file: ' + str(path))
    require(path.resolve() == path, 'no symlink components: ' + str(path))
    return path


def read(path):
    return json.loads(regular(path).read_text())


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'finite saved JSON values')
    else:
        require(value is None or isinstance(value, (str, bool, int)), 'ordinary JSON value')


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder,
            'regular absolute evidence directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no evidence symlinks')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(regular(path))
        else:
            require(path.is_dir(), 'ordinary evidence members')
    return result


def phase(plan, plan_path, name):
    spec = plan['phases'][name]
    receipt_path = Path(spec['output'] + '.receipt.json')
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['phase'] == name
            and receipt['plan'] == str(plan_path)
            and receipt['plan_sha256'] == descriptor(plan_path)['sha256']
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['parent_before'] == receipt['parent_after'] == plan['parent']
            and receipt['output'] == spec['output'] and receipt['supervision'] == spec['supervision'],
            'source-bound successful original phase')
    launch_path = Path(spec['supervision'])
    require(receipt['launch'] == read(launch_path), 'original persisted launch')
    diagnostic.validate_launch_binding(plan, plan_path, name, receipt['launch'])
    terminal_path = diagnostic.closed_producer(receipt)
    directory = Path(spec['output'])
    expected = {'command-0.log', 'command-1.log'} if name == 'qualify' else {'geometry.json'}
    require(inventory(directory) == receipt['files'] and set(receipt['files']) == expected,
            'complete original phase payload inventory')
    log = Path(str(launch_path).replace('.launch.json', '.log'))
    regular(log)
    return {'receipt': receipt, 'receipt_path': receipt_path, 'launch_path': launch_path,
            'terminal_path': terminal_path, 'terminal': read(terminal_path), 'log_path': log,
            'directory': directory}


def failed_attempt(study, current):
    """Authenticate failure as failure against its own preserved source bytes."""
    path = study / 'registration-01.json'
    require(descriptor(regular(path))['sha256'] == FAILED_REGISTRATION_SHA256, 'original failed registration')
    plan = read(path)
    require(plan['version'] == diagnostic.VERSION and plan['parent'] == current['parent']
            and plan['runtime'] == current['runtime'] and len(plan['sources']) == 52,
            'same parent and runtime in failed qualification')
    changed = {key for key in plan['sources'] | current['sources']
               if plan['sources'].get(key) != current['sources'].get(key)}
    require(changed == {'tests/test_finite_transport_geometry.py'}, 'test-only correction, unchanged diagnostic formulas')
    spec = plan['phases']['qualify']
    require(spec['output'] == str(study / 'engineering-01')
            and spec['supervision'] == str(study / 'engineering-native-01.launch.json'), 'exact first attempt paths')
    receipt = read(Path(spec['output'] + '.receipt.json'))
    launch = read(Path(spec['supervision']))
    require(receipt['status'] == 'FAILED' and receipt['phase'] == 'qualify'
            and receipt['plan'] == str(path) and receipt['plan_sha256'] == FAILED_REGISTRATION_SHA256
            and receipt['sources_before'] == plan['sources'] and receipt['parent_before'] == plan['parent']
            and receipt['launch'] == launch and receipt['output'] == spec['output']
            and receipt['supervision'] == spec['supervision'], 'original failed worker identity')
    diagnostic.validate_launch_binding(plan, path, 'qualify', launch)
    terminal = read(Path(spec['supervision'].replace('.launch.json', '.terminal.json')))
    require(terminal['status'] == 'failed' and terminal['returncode'] == 1
            and terminal['group_absent'] and terminal['cleanup']['reaped']
            and not terminal['cleanup']['errors'] and not terminal['timed_out']
            and terminal['timing_available'] and terminal['error'] is None
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'ordinary closed qualification failure, not timeout')
    for key in ('pid', 'pgid', 'command', 'cwd', 'cap_seconds', 'clock_backend', 'started_ns',
                'deadline_ns', 'watchdog_sha256', 'clock_source_sha256'):
        require(terminal[key] == launch[key], 'failed original launch/terminal join: ' + key)
    expected_commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
                         [plan['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *plan['tests']]]
    require([row['command'] for row in receipt['commands']] == expected_commands
            and [row['returncode'] for row in receipt['commands']] == [0, 1]
            and inventory(Path(spec['output'])) == receipt['files']
            and set(receipt['files']) == {'command-0.log', 'command-1.log'}, 'failed command and log identities')
    require(re.search(r'(?m)^3 failed, 50 passed in [^\n]+$',
                      (Path(spec['output']) / 'command-1.log').read_text()) is not None,
            'preserved three-failure, fifty-pass fabricated attempt')
    snapshot = study / 'engineering-01-source-snapshot'
    manifest_path = snapshot / 'manifest.json'
    manifest = read(manifest_path)
    require(manifest['registration'] == {'path': str(path), 'sha256': FAILED_REGISTRATION_SHA256}
            and manifest['sources'] == plan['sources'] and isinstance(manifest['reason'], str)
            and bool(manifest['reason']), 'failed source snapshot registration association')
    require(inventory(snapshot) == {**plan['sources'], 'manifest.json': descriptor(manifest_path)},
            'exact 52 failed sources plus the snapshot manifest')
    return {'registration': {'path': str(path), **descriptor(path)},
            'status': 'FAILED', 'tests_failed': 3, 'tests_passed': 50,
            'original_wall_seconds': terminal['wall_seconds'], 'finished_ns': terminal['finished_ns'],
            'reason': manifest['reason'],
            'correction': 'Test-only analytical float64 bounds: u=eps/2, gamma_n=n*u/(1-n*u); '
                          'normalization uses 4*gamma_n and projection allows two versus one rounded matrix products. '
                          'No fitted residual cutoff, core change or scientific threshold change.'}


def authenticate(study, registration_sha256):
    study = Path(study).resolve()
    require(study == diagnostic.FOLDER and Path.cwd() == ROOT, 'exact child directory and repository cwd')
    plan_path = study / 'registration-02.json'
    require(descriptor(regular(plan_path))['sha256'] == registration_sha256 == REGISTRATION_SHA256,
            'supplied exact successful registration digest')
    plan = read(plan_path)
    require(plan['version'] == diagnostic.VERSION and plan['root'] == str(ROOT)
            and len(plan['sources']) == 52 and plan['tests'] == diagnostic.TESTS
            and plan['lint_sources'] == diagnostic.LINT and set(plan['phases']) == {'qualify', 'diagnose'},
            'fixed diagnostic configuration')
    require(plan['phases']['qualify'] == {'output': str(study / 'engineering-02'),
            'supervision': str(study / 'engineering-native-02.launch.json'), 'cap_seconds': 90}
            and plan['phases']['diagnose'] == {'output': str(study / 'run-01'),
            'supervision': str(study / 'diagnostic-native-01.launch.json'), 'cap_seconds': 90}, 'exact original phase paths')
    sources = {}
    for name, expected in plan['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative registered source')
        path = regular(ROOT / name)
        require(descriptor(path) == expected, 'unchanged source: ' + name)
        sources[str(path)] = expected
    parent_auth = diagnostic.authenticate(diagnostic.PARENT)
    parent = diagnostic.parent_identity(parent_auth)
    require(parent == plan['parent'], 'entire original parent identity and opaque inventory unchanged')
    failed = failed_attempt(study, plan)
    qualification, qualification_terminal = diagnostic.admit_qualification(plan, plan_path)
    phases = {name: phase(plan, plan_path, name) for name in ('qualify', 'diagnose')}
    require(phases['qualify']['receipt_path'] == qualification
            and phases['qualify']['terminal_path'] == qualification_terminal
            and failed['finished_ns'] <= phases['qualify']['terminal']['started_ns']
            and phases['qualify']['terminal']['finished_ns'] <= phases['diagnose']['terminal']['started_ns'],
            'ordered original closed attempts')
    receipt = phases['diagnose']['receipt']
    require(receipt['qualification_receipt'] == descriptor(qualification)
            and receipt['qualification_terminal'] == descriptor(qualification_terminal)
            and receipt['result'] == COUNTS, 'closed diagnostic exact decode and no-call counts')
    snapshot = study / 'qualified-source-snapshot'
    snapshot_path = snapshot / 'manifest.json'
    snapshot_manifest = read(snapshot_path)
    require(snapshot_manifest['registration'] == {'path': str(plan_path), 'sha256': REGISTRATION_SHA256}
            and snapshot_manifest['sources'] == plan['sources']
            and inventory(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot_path)},
            'exact 52 qualified source copies and their associated manifest')
    child = inventory(study)
    require(all(not name.endswith('.npz') for name in child), 'child archive contains no parent checkpoint arrays')
    inputs = {str(study / name): value for name, value in child.items()}
    inputs.update(sources)
    inputs.update(parent['inputs'])
    inputs[str(Path(__file__).resolve())] = descriptor(Path(__file__).resolve())
    # All phase closures, sources and opaque inventories above precede this read.
    geometry = read(phases['diagnose']['directory'] / 'geometry.json')
    require(geometry['version'] == diagnostic.VERSION and geometry['counts'] == COUNTS
            and geometry['parent'] == parent['registration'] and geometry['architecture_claim'] is False
            and geometry['candidate_advances'] is False
            and geometry['requires_original_supervisor_closure'] is True, 'saved diagnostic claim scope')
    require(geometry['existing_endpoint_rows'] == parent_auth['audit']['rows']
            and geometry['existing_prefix_rows'] == parent_auth['audit']['prefix_rows'], 'unmodified existing error rows')
    roster = [(arm, seed, label) for arm in diagnostic.ARMS for seed in diagnostic.SEEDS for label in diagnostic.LABELS]
    require([(r['arm'], r['seed'], r['label']) for r in geometry['records']] == roster, 'complete ordered 27 snapshots')
    for row in geometry['records']:
        name = f"{row['label']}-{row['arm']}-{row['seed']}.npz"
        expected = parent_auth['summary']['files'][name]
        require(row['checkpoint'] == {'path': str(parent_auth['phases']['fit']['directory'] / name), **expected},
                'each original checkpoint identity retained')
    finite_tree(geometry)
    return {'plan': plan, 'plan_path': plan_path, 'phases': phases, 'parent': parent,
            'geometry': geometry, 'failed': failed, 'child_files': child, 'inputs': inputs}


def summarize(auth):
    result = auth['geometry']
    compact = []
    for row in result['records']:
        old = row['geometry']
        require(old['version'] == 'finite-transport-geometry-v1' and old['dtype'] == 'float64'
                and old['state_dimension'] == 8 and old['action_count'] == 4
                and len(old['transition']) == 4 and old['grams']['horizons'] == [0, 1, 2, 4, 8], 'geometry schema')
        item = {key: value for key, value in old.items() if key not in ('fields', 'basis', 'grams')}
        item['grams'] = {'horizons': old['grams']['horizons'], 'work': old['grams']['work']}
        for channel in ('cost', 'state'):
            require([g['horizon'] for g in old['grams'][channel]] == [0, 1, 2, 4, 8], 'all fixed Gram horizons')
            item['grams'][channel] = [{key: g[key] for key in ('horizon', 'eigenvalues', 'trace', 'ranks')}
                                      for g in old['grams'][channel]]
        compact.append({**{key: row[key] for key in ('arm', 'seed', 'label', 'checkpoint')}, 'geometry': item})
    log = (auth['phases']['qualify']['directory'] / 'command-1.log').read_text()
    matched = re.findall(r'(?m)^\s*(\d+) passed in [^\n]+$', log)
    require(len(matched) == 1 and int(matched[0]) == 53, 'complete successful fabricated test roster')
    return {'version': VERSION, 'records': compact, 'existing_endpoint_rows': result['existing_endpoint_rows'],
            'existing_prefix_rows': result['existing_prefix_rows'], 'diagnostic_counts': COUNTS,
            'publication_counts': PUBLICATION_COUNTS, 'failed_qualification': auth['failed'],
            'successful_qualification_tests': int(matched[0]),
            'phase_seconds': {name: record['terminal']['wall_seconds'] for name, record in auth['phases'].items()},
            'parent': auth['parent']['registration'], 'parent_release_required': PARENT_RELEASE,
            'performance_gate': None, 'candidate_advances': False, 'architecture_claim': False,
            'scope': result['scope'],
            'compact_scope': 'All scalar metrics, raw distance/singular/eigen spectra, fixed-threshold ranks, work and existing error rows retained. '
                             'Raw reconstructed fields, basis and full/projected Gram matrices remain in archived geometry.json.',
            'limits': 'Descriptive model geometry; no causal information-loss estimate or true-state alignment. '
                      'Uniform action-word sensitivity combines transport and hazard survival. '
                      'O/head rank at most three is structural. No candidate or closed H4-training follow-up is admitted.'}


def figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = [(arm, seed) for arm in diagnostic.ARMS for seed in diagnostic.SEEDS]
    rows = {(r['arm'], r['seed'], r['label']): r['geometry'] for r in summary['records']}
    errors = {(r['arm'], r['seed'], r['horizon']): r for r in summary['existing_endpoint_rows']}
    require(len(errors) == 36 and set(errors) == {(a, s, h) for a, s in keys for h in (1, 2, 4, 8)}, 'all existing 36 error rows')
    labels = [f"{arm.replace('joint_', '').replace('prefix_then_joint', 'prefix then joint')} / {seed}" for arm, seed in keys]
    panels = [
        ('Mean transition contraction (Dobrushin)',
         [[math.fsum(t['dobrushin_delta'] for t in rows[a, s, stage]['transition']) / 4
           for stage in diagnostic.LABELS] for a, s in keys], list(diagnostic.LABELS)),
        ('Mean drift of uniform mass (L1)',
         [[math.fsum(t['uniform_image_l1'] for t in rows[a, s, stage]['transition']) / 4
           for stage in diagnostic.LABELS] for a, s in keys], list(diagnostic.LABELS)),
        ('H8 cost sensitivity Gram trace',
         [[next(g['trace'] for g in rows[a, s, stage]['grams']['cost'] if g['horizon'] == 8)
           for stage in diagnostic.LABELS] for a, s in keys], list(diagnostic.LABELS)),
        ('Existing final blind regret', [[errors[a, s, h]['blind_regret'] for h in (2, 8)]
                                        for a, s in keys], ['H2', 'H8']),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (title, values, columns) in zip(axes.flat, panels, strict=True):
        image = axis.imshow(values, aspect='auto', cmap='viridis')
        axis.set_title(title, pad=12)
        axis.set_xticks(range(len(columns)), columns)
        axis.set_yticks(range(len(labels)), labels)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                axis.text(j, i, f'{value:.3g}', ha='center', va='center', color='white', fontsize=8,
                          bbox={'facecolor': 'black', 'alpha': .4, 'edgecolor': 'none', 'pad': 1})
        fig.colorbar(image, ax=axis, fraction=.04, pad=.025)
    fig.suptitle('Saved recurrent transport: descriptive synthetic-model diagnostic', fontsize=15)
    fig.text(.5, .015, 'All nine fits and all 27 stages retained. Panels use separate linear color scales. '
             'Geometry is not causal information loss or a performance gate.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .955), w_pad=3, h_pad=3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def document(summary):
    lines = ['# Saved recurrent transport diagnostic', '',
             ('This is a descriptive diagnostic of all 27 saved states from the completed equal-time synthetic study. '
              'It does not train, rerun forecasting, select a fit or advance an architecture.'), '',
             ('[Protocol](../finite-transport-diagnostic-protocol.md) · [All saved metrics](summary.json) · '
              '[Evidence manifest](manifest.json) · [Child evidence archive](evidence.tar.gz)'), '',
             '![All checkpoints and existing errors](benchmark.png)', '',
             ('The three geometry panels retain initial, boundary and final states for all nine fits. '
              'Transition quantities are arithmetic averages over four actions. The regret panel uses the original '
              'independently audited H2/H8 rows; no evaluation was repeated.'), '',
             '| Original phase | Status | Seconds |', '|---|---|---:|',
             f"| Qualification 01 | FAILED: 3 assertions, 50 passed | {summary['failed_qualification']['original_wall_seconds']:.6f} |",
             f"| Qualification 02 | PASS: {summary['successful_qualification_tests']} tests | {summary['phase_seconds']['qualify']:.6f} |",
             f"| Diagnostic | PASS: 27 checkpoints, 108 arrays | {summary['phase_seconds']['diagnose']:.6f} |", '',
             summary['failed_qualification']['correction'], '',
             ('The first failure, its complete source snapshot, original logs and receipts are retained. '
              'Only qualification 02 admitted checkpoint decoding. The diagnostic made zero model, optimizer '
              'or generator calls and created no training updates or evaluation cases.'), '',
             ('The Grams average over all uniformly weighted action sequences. They propagate surviving mass '
              'without normalization, so attenuation combines transport mixing and hazard loss. '
              'Full eight-dimensional propagation precedes projection. Small signed eigenvalues are retained. '
              'The rank-at-most-three emission and head spectra have structural zero modes, which are not learned collapse.'), '',
             ('Large sensitivity does not establish correctness, and small sensitivity does not establish lost task information. '
              'The earlier retention failure and the closed H4-supervision follow-up remain unchanged. '
              'No causal, biological, robotics, novelty or architecture-improvement claim follows from these plots.'), '',
             (f'**Dependency:** [the complete parent release]({PARENT_RELEASE}) is required for the 27 original checkpoints '
              'and their evidence. This archive contains the child study only, including raw geometry.json and both complete source snapshots. '
              'Current registered source hashes and the complete external parent identity are in the manifest and receipt. '
              'The executable environment is not bundled and this package does not authorize a rerun.'), '']
    return '\n'.join(lines)


def archive_child(auth, output):
    study = diagnostic.FOLDER
    members = {'evidence/' + name: {'path': str(study / name), **pin} for name, pin in auth['child_files'].items()}
    manifest = {'version': VERSION, 'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
                'files': members, 'manifest_excludes_itself': True, 'parent_required': auth['parent'],
                'parent_release': PARENT_RELEASE, 'current_source_pins': auth['plan']['sources'],
                'qualification_failure': auth['failed'], 'diagnostic_counts': COUNTS,
                'publication_counts': PUBLICATION_COUNTS,
                'scope': 'Complete diagnostic child folder and both source snapshots, no parent weights or datasets. Parent release and executable environment remain external.',
                'archive_metadata': 'Sorted regular members, mode0644, uid/gid0, empty owner names, timestamps0, gzip filename empty.'}
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(payload)
    archive_path = output / 'evidence.tar.gz'
    with archive_path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed, \
            tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted([*members, 'manifest.json']):
            info = tarfile.TarInfo(name)
            info.size = len(payload) if name == 'manifest.json' else members[name]['bytes']
            info.mode, info.mtime, info.uid, info.gid = 0o644, 0, 0, 0
            if name == 'manifest.json':
                archive.addfile(info, io.BytesIO(payload))
            else:
                member = members[name]
                require(descriptor(regular(Path(member['path']))) == {key: member[key] for key in ('sha256', 'bytes')},
                        'unchanged opaque member before archive')
                with Path(member['path']).open('rb') as stream:
                    archive.addfile(info, stream)
    with tarfile.open(archive_path, 'r:gz') as archive:
        entries = archive.getmembers()
        require([entry.name for entry in entries] == sorted([*members, 'manifest.json'])
                and all(entry.isfile() for entry in entries), 'exact regular archive roster')
        for entry in entries:
            expected = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)} \
                if entry.name == 'manifest.json' else members[entry.name]
            stream = archive.extractfile(entry)
            require(stream is not None, 'readable archive member')
            digest, size = hashlib.sha256(), 0
            with stream:
                for block in iter(lambda current=stream: current.read(1048576), b''):
                    digest.update(block)
                    size += len(block)
            require(digest.hexdigest() == expected['sha256'] and size == expected['bytes'],
                    'byte-identical opaque archive roundtrip')
    return len(members) + 1


def publish(study, output, registration_sha256):
    output = Path(output).resolve()
    require(output == ROOT / 'research/finite-transport-diagnostic-results' and not output.exists(),
            'exclusive exact publication output')
    auth = authenticate(study, registration_sha256)
    summary = summarize(auth)
    output.mkdir()
    diagnostic.publish(output / 'summary.json', summary)
    with (output / 'report.md').open('x') as stream:
        stream.write(document(summary))
    figure(summary, output / 'benchmark.png')
    archive_members = archive_child(auth, output)
    after = authenticate(study, registration_sha256)
    require(after == auth, 'all source, child, parent and phase evidence unchanged after publication')
    names = ('summary.json', 'report.md', 'benchmark.png', 'manifest.json', 'evidence.tar.gz')
    receipt = {'version': VERSION, 'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
               'publisher': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
               'inputs_before': auth['inputs'], 'inputs_after': after['inputs'], 'inputs_unchanged': True,
               'files': {name: descriptor(output / name) for name in names},
               'archive_members': archive_members, 'opaque_roundtrip': True,
               'counts': PUBLICATION_COUNTS, 'performance_gate': None, 'candidate_advances': False,
               'parent_release_required': PARENT_RELEASE}
    diagnostic.publish(output / 'receipt.json', receipt)
    return {'output': str(output), 'receipt': descriptor(output / 'receipt.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, args.registration_sha256)), flush=True)


if __name__ == '__main__':
    main()
