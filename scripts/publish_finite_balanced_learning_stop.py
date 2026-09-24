"""Publish one failed integration qualification from authenticated metadata only.

No learner, primitive, array decoder, generator or optimizer is imported or
replayed. The source-pinned prerequisite publisher authenticates only its own
standard-library metadata. All scientific paths must remain absent. Temporary
engineering artifacts are opaque copies bound to a separate preservation receipt.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import math
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'finite-balanced-learning-v1'
VERSION = 'finite-balanced-learning-stopped-publication-v1'
REGISTRATION_SHA256 = '30d1662d115b216a35b2064ce3d39ffd5d4bc0c3cd789ffd1f8390ca70a54a0c'
CLOSURE = {
    'engineering-01.receipt.json': '38bdc68a25b93c6e349d82499c9cb22f7c5f9470c4f9f20c75cf18e151652c1a',
    'engineering-native-01.launch.json': 'ca989d0b9e9100bc71b1e45be0e05e29d55f416263f9ad71008b4bbaa35168b1',
    'engineering-native-01.terminal.json': '1cbc88e601314999103c89964103c4eed7f02ff2c7af3b638bcfe4ce04911fa9',
}
CONFIG = {'seed_namespace': 431260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003, 'fit_seeds': [431261001, 431261002, 431261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'stage1_seconds': 10., 'total_seconds': 40.}
COUNTS = dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'optimizer_calls',
                       'generator_calls', 'native_calls', 'qualification_reruns'), 0)
OUTPUT = ROOT / 'research/finite-balanced-learning-stop-results'
REPORT = ROOT / 'research/finite-balanced-learning-stop-results.md'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
            and path.is_file(), 'ordinary absolute file without symlink components: ' + str(path))
    return path

def byte_descriptor(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}

def descriptor(path):
    return byte_descriptor(regular(path).read_bytes())

def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'finite saved JSON')

def read(path):
    value = json.loads(regular(path).read_text())
    finite_tree(value)
    return value

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()

def write(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def inventory(folder):
    require(folder.is_absolute() and folder.resolve() == folder
            and folder.is_dir() and not folder.is_symlink(), 'ordinary evidence directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no evidence symlinks')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(path)
        else:
            require(path.is_dir(), 'regular files and directories only')
    return result

def archive(path, payloads):
    """Deterministic regular members followed by opaque, byte-exact verification."""
    with path.open('xb') as raw:
        with (gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed,
              tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as bundle):
            for name, payload in sorted(payloads.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(payload), 0o644, 0
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                bundle.addfile(member, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    with tarfile.open(path, 'r:gz') as bundle:
        members = bundle.getmembers()
        require([member.name for member in members] == sorted(payloads), 'exact sorted archive roster')
        for member in members:
            require(member.isfile() and member.mode == 0o644 and member.mtime == 0
                    and member.uid == member.gid == 0 and member.uname == member.gname == '',
                    'deterministic ordinary archive members')
            with bundle.extractfile(member) as stream:
                require(stream.read() == payloads[member.name], 'byte-identical opaque archive roundtrip')
    return {'verified': True, 'members': len(payloads), 'bytes': sum(map(len, payloads.values()))}


def safe_relative(name):
    require(type(name) is str and name and '\\' not in name, 'portable relative member name')
    require(not PurePosixPath(name).is_absolute()
            and all(piece not in ('', '.', '..') for piece in name.split('/')), 'safe relative member')
    return name


def authenticate(study, registration_sha256, preservation_sha256):
    study = Path(study)
    require(study == ROOT / 'output' / STUDY and Path.cwd() == ROOT, 'exact original study and cwd')
    plan_path = regular(study / 'engineering-registration-01.json')
    require(descriptor(plan_path)['sha256'] == registration_sha256 == REGISTRATION_SHA256,
            'externally supplied exact failed engineering registration')
    plan = read(plan_path)
    require(plan['version'] == STUDY and plan['mode'] == 'engineering' and plan['root'] == str(ROOT)
            and plan['config'] == CONFIG and len(plan['sources']) == 63
            and plan['rss_limit_bytes'] == 4 * 1024**3 and plan['output_limit_bytes'] == 512 * 1024**2,
            'original engineering-only registration and complete source closure')
    require({name: value['cap_seconds'] for name, value in plan['phases'].items()}
            == {'qualify': 300, 'fit': 1200, 'audit': 600}, 'original registered phase limits')
    inputs = {str(plan_path): descriptor(plan_path)}
    for name, pin in plan['sources'].items():
        path = ROOT / safe_relative(name)
        require(descriptor(path) == pin, 'unchanged original source: ' + name)
        inputs[str(path)] = pin
    snapshot = study / 'source-snapshot-engineering-01'
    snapshot_manifest = snapshot / 'manifest.json'
    require(read(snapshot_manifest) == {'registration': {'path': str(plan_path), **descriptor(plan_path)},
                                       'sources': plan['sources']}, 'original snapshot manifest identity')
    snapshot_files = inventory(snapshot)
    require(snapshot_files == {**plan['sources'], 'manifest.json': descriptor(snapshot_manifest)},
            'exact complete source snapshot including manifest')
    inputs.update({str(snapshot / name): pin for name, pin in snapshot_files.items()})
    spec = plan['phases']['qualify']
    phase, launch_path = Path(spec['output']), Path(spec['supervision'])
    require(phase == study / 'engineering-01' and launch_path == study / 'engineering-native-01.launch.json',
            'original qualification paths')
    for name, sha in CLOSURE.items():
        require(descriptor(study / name)['sha256'] == sha, 'fixed original closed evidence: ' + name)
    receipt_path = study / 'engineering-01.receipt.json'
    terminal_path, log_path = study / 'engineering-native-01.terminal.json', study / 'engineering-native-01.log'
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['phase'] == 'qualify' and receipt['status'] == 'FAILED'
            and receipt['error'] == "ValueError('qualification command 1 failed')"
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == registration_sha256
            and receipt['output'] == str(phase) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == plan['sources'] and 'sources_after' not in receipt
            and 'result' not in receipt and receipt['launch'] == launch,
            'original failed receipt, without a fabricated successful source-after record')
    expected_command = [plan['runtime']['executable'], str(ROOT / 'scripts/finite_balanced_learning_worker.py'),
        '--plan', str(plan_path), '--plan-sha256', registration_sha256, '--phase', 'qualify',
        '--supervision', str(launch_path), '--output', str(phase)]
    require(launch['command'] == expected_command and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == 300 and launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'original exact interpreter, worker, arguments, cwd, deadline and native source binding')
    require(all(type(launch[k]) is int and launch[k] > 0 for k in
                ('pid', 'pgid', 'parent_pid', 'started_ns', 'deadline_ns'))
            and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
            and launch['deadline_ns'] - launch['started_ns'] == 300 * 10**9,
            'original process group and deadline')
    require(all(terminal.get(k) == value for k, value in launch.items()), 'full original launch/terminal join')
    cleanup = terminal['cleanup']
    require(terminal['status'] == 'failed' and terminal['returncode'] == 1
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['timing_available'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and cleanup['reaped'] is True and cleanup['group_absent'] is True
            and cleanup['errors'] == [] and cleanup['signals'] == [], 'failed child cleanly reaped, no timeout')
    require(launch['started_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 10**9
            and launch['started_unix'] <= receipt['started_unix'] <= receipt['finished_unix'] <= terminal['finished_unix'],
            'original bounded process timing')
    phase_files = inventory(phase)
    require(phase_files == receipt['files'] and set(phase_files) == {'command-0.log', 'command-1.log'},
            'complete unchanged original command logs')
    commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']],
                [plan['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 '--noconftest', *plan['tests']]]
    require(len(receipt['commands']) == 2, 'exact two completed qualification commands')
    for row, command, exit_code in zip(receipt['commands'], commands, (0, 1), strict=True):
        require(row['command'] == command and row['returncode'] == exit_code
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds'])
                and 0 <= row['seconds'] <= terminal['wall_seconds'], 'original lint-pass/test-failure commands')
    require(sum(row['seconds'] for row in receipt['commands']) <= terminal['wall_seconds'],
            'commands nested inside original process time')
    for path in (receipt_path, launch_path, terminal_path, log_path):
        inputs[str(path)] = descriptor(path)
    inputs.update({str(phase / name): pin for name, pin in phase_files.items()})
    # The prerequisite proof is authenticated by its already-pinned stdlib helper.
    # No kernel or numerical library is imported by this metadata path.
    from publish_finite_balanced_transition_qualification import REGISTRATION_SHA256 as kernel_sha
    from publish_finite_balanced_transition_qualification import authenticate as authenticate_kernel
    proof = plan['kernel_qualification']
    kernel_folder = ROOT / 'output/finite-balanced-transition-qualification-v1'
    require(Path(proof['folder']) == kernel_folder and inventory(kernel_folder) == proof['files'],
            'complete unchanged prerequisite evidence')
    prerequisite = authenticate_kernel(kernel_folder, kernel_sha)
    require(all(plan['sources'].get(name) == pin for name, pin in prerequisite['plan']['sources'].items()),
            'same qualified primitive sources in failed integration')
    inputs.update({str(kernel_folder / name): pin for name, pin in proof['files'].items()})
    preservation_path = regular(study / 'manifest-preserved-engineering-01.json')
    require(re.fullmatch('[0-9a-f]{64}', preservation_sha256) is not None
            and descriptor(preservation_path)['sha256'] == preservation_sha256, 'external preservation-manifest pin')
    preservation = read(preservation_path)
    require(preservation['preserved_root'] == str(study / 'preserved-engineering-01')
            and Path(preservation['original_root']).is_absolute()
            and preservation['copied_unix'] >= terminal['finished_unix'],
            'explicit post-closure preservation roots and timestamp')
    items = preservation['items']
    require(type(items) is dict and items, 'nonempty explicit preservation inventory')
    preserved = study / 'preserved-engineering-01'
    expected = {}
    original_paths = set()
    for name, item in items.items():
        safe_relative(name)
        require(type(item) is dict and set(item) == {'original_path', 'sha256', 'bytes'}
                and type(item['original_path']) is str and Path(item['original_path']).is_absolute()
                and item['original_path'] == str(Path(preservation['original_root']) / name)
                and item['original_path'] not in original_paths, 'one original absolute path for each preserved copy')
        original_paths.add(item['original_path'])
        pin = {'sha256': item['sha256'], 'bytes': item['bytes']}
        require(descriptor(preserved / name) == pin, 'unchanged opaque preserved artifact: ' + name)
        expected[name] = pin
    require(inventory(preserved) == expected, 'complete exact preserved engineering tree')
    inputs[str(preservation_path)] = descriptor(preservation_path)
    inputs.update({str(preserved / name): pin for name, pin in expected.items()})
    require(not (study / 'study-registration.json').exists(), 'scientific study was never registered')
    for name in ('fit', 'audit'):
        phase_spec = plan['phases'][name]
        output = Path(phase_spec['output'])
        launch = Path(phase_spec['supervision'])
        for path in (output, Path(str(output) + '.receipt.json'), launch,
                     launch.with_name(launch.name.removesuffix('.launch.json') + '.terminal.json'),
                     launch.with_name(launch.name.removesuffix('.launch.json') + '.log')):
            require(not path.exists(), 'no scientific phase or admission artifact: ' + str(path))
    # Only after source, closure and inventory authentication interpret the logs.
    require((phase / 'command-0.log').read_text() == 'All checks passed!\n', 'original successful lint log')
    test_log = (phase / 'command-1.log').read_text()
    match = re.search(r'^1 failed, 204 passed, 1 warning in ([0-9]+(?:\.[0-9]+)?)s\s*$', test_log, re.MULTILINE)
    failed = re.findall(r'^FAILED (\S+)', test_log, re.MULTILINE)
    require(match is not None and failed == [
        'tests/test_finite_balanced_learning_runner.py::test_three_transition_models_complete_before_fresh_dev_and_independent_audit'],
        'exact original integration failure and complete pytest totals')
    residuals = re.findall(r'ValueError: fixed-sweep residual exceeded tolerance: row=([0-9.e+-]+), '
                          r'column=([0-9.e+-]+), tolerance=([0-9.e+-]+), sweeps=([0-9]+)', test_log)
    require(len(residuals) == 1, 'one logged actual fixed-sweep guard failure')
    row, column, tolerance = map(float, residuals[0][:3])
    sweeps = int(residuals[0][3])
    require(row == 1.014077710692618e-12 and column == 2.220446049250313e-16
            and tolerance == 1e-12 and sweeps == 64 and row > tolerance, 'unchanged logged numerical stop')
    failure = read(preserved / 'failure.json')
    allocation = read(preserved / 'allocation-balanced-940101.json')
    fit_rows = [json.loads(line) for line in regular(preserved / 'fits.jsonl').read_text().splitlines()]
    require(failure['version'] == STUDY and failure['stage'] == 'fit balanced 940101'
            and failure['completed_fits'] == failure['counts']['fit_count'] == 2
            and [(r['arm'], r['seed']) for r in fit_rows] == [('original_free', 940101), ('matched_free', 940101)],
            'two completed engineering free-control fits followed by balanced failure')
    zero_fields = ('dev_generation_count', 'evaluation_blind_rollouts', 'evaluation_observed_rollouts',
                   'evaluation_shuffled_rollouts', 'evaluation_case_views', 'evaluation_prefix_rollouts',
                   'evaluation_prefix_event_views', 'external_model_calls', 'native_calls', 'teacher_calls')
    require(all(failure['counts'][name] == 0 for name in zero_fields),
            'producer-attested engineering stop before development generation and evaluation')
    trace = allocation['trace']
    require(allocation['version'] == 'finite-training-allocation-v1'
            and allocation['arm'] == 'prefix_then_joint' and allocation['status'] == 'FAILED_EXCEPTION'
            and allocation['error'] == failure['error'] and allocation['joint_cursor'] == 0
            and allocation['stages'] == [] and len(trace) == 300,
            'original balanced engineering exception within first stage')
    require(all(r['attempt'] == index + 1 and r['kind'] == 'prefix' and r['stage'] == 1
                and r['accepted'] is True and r['rolled_back'] is False and r['error'] is None
                for index, r in enumerate(trace[:-1])), '299 accepted prefix updates in original controller log')
    last = trace[-1]
    require(last['attempt'] == 300 and last['kind'] == 'prefix' and last['stage'] == 1
            and last['accepted'] is False and last['rolled_back'] is True
            and last['error'] == failure['error'] and last['completed_elapsed'] is None
            and last['model_retained_sha256'] == last['model_before_sha256']
            and last['optimizer_retained_sha256'] == last['optimizer_before_sha256']
            and last['cursor_retained'] == last['cursor_before'] == 0,
            'failed attempt records model, optimizer and cursor rollback without a completion time')
    partial = {'completed_free_fits': 2, 'accepted_balanced_prefix_updates': 299,
               'failed_balanced_attempt': 300, 'failed_attempt_start_elapsed': last['start_elapsed'],
               'failed_attempt_completion_elapsed': None, 'rollback_logged': True,
               'development_generation_count': 0, 'development_evaluation_count': 0,
               'producer_failure_counts': failure['counts'], 'numerical_states_replayed': False}
    source = Path(__file__).resolve()
    inputs[str(source)] = descriptor(source)
    return {'plan': plan, 'plan_path': plan_path, 'receipt': receipt, 'terminal': terminal,
            'inputs': inputs, 'preservation': preservation, 'preserved_files': len(items),
            'pytest_seconds': float(match.group(1)), 'engineering_partial': partial, 'failure': {'row_residual': row,
                'column_residual': column, 'tolerance': tolerance, 'sweeps': sweeps},
            'prerequisite_registration': {'path': str(prerequisite['plan_path']),
                                         **descriptor(prerequisite['plan_path'])}}


def summarize(auth):
    return {'version': VERSION, 'study': STUDY, 'status': 'QUALIFICATION_FAILED',
            'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
            'lint_passed': True, 'tests_passed': 204, 'tests_failed': 1, 'test_warnings': 1,
            'original_process_returncode': 1, 'original_wall_seconds': auth['terminal']['wall_seconds'],
            'original_timing_scope': auth['terminal']['timing_scope'], 'pytest_seconds': auth['pytest_seconds'],
            'source_files': len(auth['plan']['sources']), 'preserved_engineering_files': auth['preserved_files'],
            'preservation_scope': 'Post-stop copies, not original worker inventory attestations. Original temporary paths remain in their manifest.',
            'failure': auth['failure'], 'engineering_partial': auth['engineering_partial'],
            'prerequisite': {'status': 'QUALIFICATION_PASS', 'registration': auth['prerequisite_registration'],
                             'scope': 'Earlier primitive-only fabricated qualification; not an integration success.'},
            'scientific_registered': False, 'scientific_started': False, 'scientific_result': None,
            'advancement_eligible': False, 'publication_counts': COUNTS,
            'interpretation': 'The fixed64 routine exceeded its frozen row-residual tolerance during the engineering training smoke. The guard stopped the attempt. This does not measure task accuracy or architectural superiority.',
            'limitations': ['Partial training and rollback fields are source-bound saved records, not an independent numerical replay.',
                           'The failed attempt has a start timestamp but no completion timestamp; start time is not failure duration.',
                           'Incomplete producer counters and the balanced partial trace have different completion scopes.',
                           'No threshold relaxation, extra sweeps, scientific rerun or follow-up admission is conferred by publication.']}


def figure(summary):
    failure = summary['failure']
    ratio = failure['row_residual'] / failure['tolerance']
    x0, width, maximum = 100., 650., 1.1
    threshold_x = x0 + width / maximum
    residual_x = x0 + width * ratio / maximum
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="250" viewBox="0 0 850 250">'
        '<rect width="850" height="250" fill="white"/>'
        '<g font-family="Arial,sans-serif" fill="#17202a">'
        '<text x="35" y="36" font-size="22">Engineering qualification stopped</text>'
        '<text x="35" y="66" font-size="15">204 tests passed; 1 integration test failed. No scientific result.</text>'
        f'<line x1="{x0}" y1="125" x2="{x0+width}" y2="125" stroke="#ccd1d1" stroke-width="12"/>'
        f'<line x1="{threshold_x}" y1="96" x2="{threshold_x}" y2="149" stroke="#17202a" stroke-width="2"/>'
        f'<circle cx="{residual_x}" cy="125" r="8" fill="#b03a2e"/>'
        '<text x="35" y="178" font-size="15">Row residual / allowed tolerance (zero-origin scale)</text>'
        f'<text x="35" y="208" font-size="15">Observed: {html.escape(format(failure["row_residual"], ".17g"))}; '
        f'limit: 1e-12; ratio: {ratio:.9f}. Fixed sweeps: 64.</text>'
        f'<text x="{x0}" y="153" font-size="12">0</text>'
        f'<text x="{threshold_x-35}" y="89" font-size="12">limit = 1</text>'
        '</g></svg>\n').encode()


def report_text(summary):
    failure, partial = summary['failure'], summary['engineering_partial']
    return ('# Balanced-transition integration stopped at qualification\n\n'
        '**QUALIFICATION_FAILED.** Lint passed; 204 tests passed and one integration test failed '
        '(one test warning). The original supervised qualification closed with exit code 1 in '
        f'{summary["original_wall_seconds"]:.9f} seconds. No scientific study was registered or started.\n\n'
        '![Original engineering stop](status.svg)\n\n'
        '| Recorded check | Original value |\n| --- | ---: |\n'
        f'| Fixed normalization sweeps | {failure["sweeps"]} |\n'
        f'| Allowed row/column residual | {failure["tolerance"]:.17g} |\n'
        f'| Observed row residual | {failure["row_residual"]:.17g} |\n'
        f'| Observed column residual | {failure["column_residual"]:.17g} |\n'
        '| Completed engineering free-control fits | 2 |\n'
        '| Accepted balanced prefix updates | 299 |\n'
        '| Failed balanced prefix attempt | 300 |\n'
        '| Engineering DEV generations and evaluations | 0 |\n\n'
        'The row residual exceeded the frozen 1e-12 limit. The exception occurred in the '
        'balanced arm of the three-arm engineering smoke, before development generation and '
        'saved-output auditing. Its controller records model, optimizer and cursor rollback. '
        f'The failed update began {partial["failed_attempt_start_elapsed"]:.9f} seconds after that fit started; '
        'it has no completion timestamp. These are preserved producer records, not a numerical replay.\n\n'
        'The earlier primitive qualification remains a separate pass. This integration failure shows '
        'that its fixed64 routine did not reliably meet the declared tolerance during this smoke. '
        'It does not establish task performance, architectural superiority or a general impossibility. '
        'The original attempt remains closed, with no added sweeps or relaxed tolerance.\n\n'
        f'The evidence archive includes all {summary["source_files"]} registered source copies, '
        f'the original process and test logs, all {summary["preserved_engineering_files"]} post-stop '
        'engineering artifact copies and the complete pinned primitive prerequisite. Original temporary '
        'paths and the separate preservation status are retained. Arrays and checkpoints are opaque. '
        'Interpreter and installed packages remain external.\n\n'
        '[Complete summary](summary.json) · [Exact member manifest](manifest.json) · '
        '[Evidence archive](evidence.tar.gz)\n').encode()


def publish(study, output, registration_sha256, preservation_sha256):
    study, output = Path(study), Path(output)
    require(output == OUTPUT and output.parent.resolve() == output.parent and not output.exists()
            and not output.is_symlink() and not REPORT.exists() and not REPORT.is_symlink(),
            'exclusive prescribed output folder and overview path')
    auth = authenticate(study, registration_sha256, preservation_sha256)
    summary = summarize(auth)
    report = report_text(summary)
    overview = report.decode().replace('(status.svg)', '(finite-balanced-learning-stop-results/status.svg)')
    for name in ('summary.json', 'manifest.json', 'evidence.tar.gz'):
        overview = overview.replace('(' + name + ')', '(finite-balanced-learning-stop-results/' + name + ')')
    outputs = {'summary.json': json_bytes(summary), 'report.md': report, 'status.svg': figure(summary)}
    payloads = {}
    for path_text, pin in auth['inputs'].items():
        path = regular(Path(path_text))
        require(path.is_relative_to(ROOT), 'all archive inputs have repository-relative destinations')
        name = 'evidence/' + safe_relative(path.relative_to(ROOT).as_posix())
        payload = path.read_bytes()
        require(byte_descriptor(payload) == pin and name not in payloads, 'unique unchanged opaque input')
        payloads[name] = payload
    for name, payload in outputs.items():
        payloads['publication/' + name] = payload
    payloads['publication/finite-balanced-learning-stop-results.md'] = overview.encode()
    manifest = {'version': VERSION, 'status': 'QUALIFICATION_FAILED', 'registration': summary['registration'],
                'preservation': {'path': str(study / 'manifest-preserved-engineering-01.json'),
                                 **descriptor(study / 'manifest-preserved-engineering-01.json')},
                'sources': auth['plan']['sources'], 'inputs': auth['inputs'],
                'files': {name: byte_descriptor(value) for name, value in sorted(payloads.items())},
                'manifest_coverage': 'Every other archive member; MANIFEST.json equals this external manifest and excludes itself.',
                'scope': 'Original failed engineering evidence, full source snapshot, explicit post-stop artifact copies, pinned prerequisite proof and this metadata-only publication.',
                'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, timestamps0, empty gzip filename.',
                'runtime': auth['plan']['runtime'], 'portable_environment_included': False,
                'scientific_registered': False, 'scientific_started': False, 'publication_counts': COUNTS}
    manifest_payload = json_bytes(manifest)
    payloads['MANIFEST.json'] = manifest_payload
    output.mkdir()
    for name, payload in outputs.items():
        write(output / name, payload)
    write(output / 'manifest.json', manifest_payload)
    roundtrip = archive(output / 'evidence.tar.gz', payloads)
    after = authenticate(study, registration_sha256, preservation_sha256)
    require(after == auth, 'all sources, snapshots, prerequisite and preserved evidence unchanged after publication')
    write(REPORT, overview.encode())
    files = {name: descriptor(output / name) for name in (*outputs, 'manifest.json', 'evidence.tar.gz')}
    receipt = {'version': VERSION, 'status': 'PASS', 'scientific_status': 'QUALIFICATION_FAILED',
               'publisher': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
               'registration': summary['registration'], 'output': str(output),
               'overview': {'path': str(REPORT), **descriptor(REPORT)}, 'files': files,
               'inputs_before': auth['inputs'], 'inputs_after': after['inputs'], 'inputs_unchanged': True,
               'opaque_archive_roundtrip': roundtrip, 'publication_counts': COUNTS,
               'scientific_registered': False, 'scientific_started': False}
    write(output / 'receipt.json', json_bytes(receipt))
    write(output / 'SHA256SUMS', ''.join(descriptor(output / name)['sha256'] + '  ' + name + '\n'
          for name in ('evidence.tar.gz', 'manifest.json', 'receipt.json')).encode())
    return {'output': str(output), 'report': str(REPORT), 'receipt': descriptor(output / 'receipt.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--preservation-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, args.registration_sha256, args.preservation_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
