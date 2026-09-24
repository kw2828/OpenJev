"""Publish one closed fabricated qualification using standard-library metadata.

This helper never imports the primitive, numerical packages, a model or a data
decoder. The archived environment description is not a portable environment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_VERSION = 'finite-joint-reuse-qualification-v1'
VERSION = 'finite-joint-reuse-qualification-publication-v1'
REGISTRATION_SHA256 = '9f0ce32aef4ae456a88bba8d79b2073404e053ac6fba4a9c54bed219deb7e6e0'
ORIGINAL_CLOSURE = {
    'engineering-01.receipt.json': 'ac51fd781492b7ad54a9009ce072b9f6523a30d37feb4bd763cd27e9c86fc145',
    'native-01.launch.json': '5c8c2f4176eaae13e6035f58bc309772a6482fa8e9680c5f322a13d632b291fb',
    'native-01.terminal.json': 'a52336e6d40e5460138289ae3d8488fef48eb11381948576d3fca178da3dce72',
    'native-01.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}
SOURCES = (
    'src/openjev/research/finite_joint_reuse.py',
    'tests/test_finite_joint_reuse.py',
    'scripts/qualify_finite_joint_reuse.py',
    'research/finite-joint-reuse-qualification-protocol.md',
    'scripts/supervise_dialogue_observation_v2.py',
    'src/openjev/research/suspend_clock.py',
    'src/openjev/__init__.py', 'src/openjev/research/__init__.py',
    'pyproject.toml', 'uv.lock',
)
NEW_SOURCES = SOURCES[:4] + ('scripts/publish_finite_update_learning_stop.py',)
STOP_REGISTRATION_SHA256 = 'df564a228c92291b7a3149037ee1bf94d3f03f6f289564dc21d40a6e9681bec5'
STOP_PUBLICATION_SHA256 = 'a9e386ed1da8f9a34dd51d9a7e6a10ed0efc5fd0ad72f6188c7fb71d4ab8d3d9'
ENVIRONMENT = {key: '1' for key in (
    'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
)}
ENVIRONMENT.update(PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
PUBLICATION_COUNTS = {
    'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0,
    'optimizer_calls': 0, 'generator_calls': 0, 'numerical_qualification_reruns': 0,
}


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


def current_runtime():
    return {'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'numpy', 'pytest', 'ruff')}}


def authenticate_prior(plan):
    """Trust the pinned stop publication, then recheck all of its opaque inputs."""
    folder = ROOT / 'output/finite-update-learning-v1'
    publication = ROOT / 'research/finite-update-learning-stop-results'
    binding = plan['prior_stop']
    require(binding['folder'] == str(folder) and binding['publication'] == str(publication)
            and inventory(folder) == binding['files']
            and inventory(publication) == binding['publication_files'],
            'complete original stop and publication inventories bound before qualification')
    old_path = folder / 'engineering-registration-01.json'
    pub_path = publication / 'receipt.json'
    require(descriptor(old_path)['sha256'] == STOP_REGISTRATION_SHA256
            and descriptor(pub_path)['sha256'] == STOP_PUBLICATION_SHA256,
            'exact original stop registration and authenticated publication receipt')
    old, published = read(old_path), read(pub_path)
    require(old['version'] == 'finite-update-learning-v1' and old['mode'] == 'engineering'
            and old['root'] == str(ROOT) and len(old['sources']) == 75,
            'original75-source engineering registration')
    require(published['status'] == 'PASS' and published['scientific_status'] == 'QUALIFICATION_FAILED'
            and published['scientific_registered'] is False and published['scientific_started'] is False
            and published['inputs_unchanged'] is True
            and published['inputs_before'] == published['inputs_after']
            and binding['publication_files'] == {**published['files'], 'receipt.json': descriptor(pub_path)},
            'publication succeeds only in preserving the unchanged original failure')
    inputs = {}
    for name, pin in published['inputs_before'].items():
        path = regular(Path(name))
        require(path.is_relative_to(ROOT) and descriptor(path) == pin,
                'every original publication input remains unchanged')
        inputs[str(path)] = pin
    for base, roster in ((folder, binding['files']), (publication, binding['publication_files'])):
        inputs.update({str(base / name): pin for name, pin in roster.items()})
    old_overview = published['overview']
    overview_path = ROOT / 'research/finite-update-learning-stop-results.md'
    require(old_overview == {'path': str(overview_path), **descriptor(overview_path)},
            'prior overview remains receipt-bound')
    inputs[str(overview_path)] = descriptor(overview_path)
    require(all(descriptor(ROOT / name) == pin for name, pin in old['sources'].items()),
            'all75 original current sources remain unchanged')
    old_snapshot = folder / 'source-snapshot-engineering-01'
    require(read(old_snapshot / 'manifest.json') == {
        'registration': {'path': str(old_path), **descriptor(old_path)}, 'sources': old['sources']},
        'original stop snapshot manifest')
    require(inventory(old_snapshot) == {
        **old['sources'], 'manifest.json': descriptor(old_snapshot / 'manifest.json')},
        'complete original75-source snapshot')
    for name in ('study-registration.json', 'run-01', 'run-01.receipt.json',
                 'fit-native-01.launch.json', 'fit-native-01.terminal.json', 'fit-native-01.log',
                 'audit-01', 'audit-01.receipt.json', 'audit-native-01.launch.json',
                 'audit-native-01.terminal.json', 'audit-native-01.log'):
        require(not (folder / name).exists(), 'failed study remains scientifically unopened')
    receipt = read(folder / 'engineering-01.receipt.json')
    launch = read(folder / 'engineering-native-01.launch.json')
    terminal = read(folder / 'engineering-native-01.terminal.json')
    require(receipt['status'] == 'FAILED' and receipt['launch'] == launch
            and receipt['plan'] == str(old_path) and receipt['plan_sha256'] == STOP_REGISTRATION_SHA256
            and receipt['sources_before'] == old['sources']
            and [row['returncode'] for row in receipt['commands']] == [0, 0, 1]
            and inventory(folder / 'engineering-01') == receipt['files'],
            'original failed qualification receipt and complete retained payloads')
    require(all(terminal.get(key) == value for key, value in launch.items())
            and terminal['status'] == 'failed' and terminal['returncode'] == 1
            and terminal['timed_out'] is False and terminal['timing_available'] is True
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
            and terminal['cleanup']['signals'] == []
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'original failed exit, no timeout, clean native process closure')
    # These inventories are already opaque inputs to the pinned publication.
    kernel = old['kernel_qualification']
    require(inventory(Path(kernel['folder'])) == kernel['files']
            and inventory(Path(kernel['publication'])) == kernel['publication_files'],
            'complete bound prior primitive and publication inventories')
    require(all(descriptor(ROOT / name) == pin for name, pin in old['parent_evidence'].items()),
            'unchanged five-file rounded-learning parent evidence')
    return {'sources': old['sources'], 'inputs': inputs, 'summary': {
        'status': 'QUALIFICATION_FAILED', 'registration_sha256': STOP_REGISTRATION_SHA256,
        'publication_receipt_sha256': STOP_PUBLICATION_SHA256,
        'study_files': len(binding['files']), 'publication_files': len(binding['publication_files']),
        'command_exit_codes': [0, 0, 1], 'original_sources': 75,
        'scientific_registered': False, 'scientific_started': False,
        'scope': 'Historical evidence copied only; no array decoding or numerical replay.',
    }}


def authenticate(study, registration_sha256):
    """Validate original process and all bytes before interpreting test logs."""
    study = Path(study).absolute()
    require(study == ROOT / 'output' / STUDY_VERSION and Path.cwd() == ROOT,
            'exact original study and repository cwd')
    plan_path = study / 'registration-01.json'
    require(descriptor(plan_path)['sha256'] == registration_sha256 == REGISTRATION_SHA256,
            'exact supplied original registration SHA256')
    plan = read(plan_path)
    prior = authenticate_prior(plan)
    phase = study / 'engineering-01'
    snapshot = study / 'source-snapshot-01'
    launch_path = study / 'native-01.launch.json'
    terminal_path = study / 'native-01.terminal.json'
    receipt_path = study / 'engineering-01.receipt.json'
    require(plan['version'] == STUDY_VERSION and plan['root'] == str(ROOT)
            and plan['output'] == str(phase) and plan['snapshot'] == str(snapshot)
            and plan['supervision'] == str(launch_path) and plan['cap_seconds'] == 180
            and plan['output_limit_bytes'] == 8 * 1024**2
            and plan['environment'] == ENVIRONMENT and plan['runtime'] == current_runtime(),
            'original fixed paths, limits, environment and current runtime')
    require(set(plan['sources']) == set(prior['sources']) | set(NEW_SOURCES) and len(plan['sources']) == 80,
            'exact parent75 plus five-source closure')
    for name in plan['sources']:
        require(descriptor(ROOT / name) == plan['sources'][name], 'unchanged current source: ' + name)
    snapshot_manifest = read(snapshot / 'manifest.json')
    require(snapshot_manifest == {
        'registration': {'path': str(plan_path), **descriptor(plan_path)}, 'sources': plan['sources'],
    }, 'snapshot manifest associated with original registration')
    require(inventory(snapshot) == {**plan['sources'], 'manifest.json': descriptor(snapshot / 'manifest.json')},
            'all80 original source copies plus snapshot manifest')
    commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *SOURCES[:3]],
                [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 '--noconftest', SOURCES[1]]]
    require(plan['commands'] == commands, 'exact two registered qualification commands')
    for name, digest in ORIGINAL_CLOSURE.items():
        require(descriptor(study / name)['sha256'] == digest, 'original closed process bytes: ' + name)
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['version'] == STUDY_VERSION and receipt['status'] == 'PASS'
            and 'error' not in receipt and receipt['plan'] == str(plan_path)
            and receipt['plan_sha256'] == registration_sha256
            and receipt['output'] == str(phase) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and receipt['launch'] == launch, 'original successful worker receipt and source joins')
    expected = [sys.executable, str(ROOT / SOURCES[2]), 'worker', '--plan', str(plan_path),
                '--plan-sha256', registration_sha256, '--supervision', str(launch_path), '--output', str(phase)]
    require(launch['version'] == 'dialogue-observation-supervision-v2'
            and launch['command'] == expected and launch['cwd'] == str(ROOT)
            and launch['cap_seconds'] == 180
            and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['watchdog_sha256'] == plan['sources'][SOURCES[4]]['sha256']
            and launch['clock_source_sha256'] == plan['sources'][SOURCES[5]]['sha256'],
            'exact original argv, cwd, cap and native clock source binding')
    require(all(type(launch[key]) is int and launch[key] > 0
                for key in ('pid', 'pgid', 'parent_pid', 'started_ns', 'deadline_ns'))
            and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
            and launch['deadline_ns'] - launch['started_ns'] == 180_000_000_000,
            'original process group and native deadline')
    require(all(terminal.get(key) == value for key, value in launch.items()),
            'complete original launch/terminal join')
    cleanup = terminal['cleanup']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timing_available'] is True and terminal['timed_out'] is False
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['group_absent'] is True and cleanup['group_absent'] is True
            and cleanup['reaped'] is True and cleanup['errors'] == [] and cleanup['signals'] == [],
            'original successful closure, no timeout, reaped child and clean process group')
    require(type(terminal['finished_ns']) is int and type(terminal['elapsed_ns']) is int
            and launch['started_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and terminal['elapsed_ns'] == terminal['finished_ns'] - launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1_000_000_000,
            'consistent original native elapsed time including group cleanup')
    require(launch['started_unix'] <= receipt['started_unix'] <= receipt['finished_unix']
            <= terminal['finished_unix'], 'original civil timestamp ordering')
    files = inventory(phase)
    require(files == receipt['files'] and set(files) == {'command-0.log', 'command-1.log'}
            and sum(pin['bytes'] for pin in files.values()) <= plan['output_limit_bytes'],
            'complete bounded command log inventory')
    require(len(receipt['commands']) == 2, 'exact two completed commands')
    for index, row in enumerate(receipt['commands']):
        require(row['command'] == commands[index] and row['returncode'] == 0
                and row['log'] == files[f'command-{index}.log']
                and type(row['elapsed_ns']) is int and 0 <= row['elapsed_ns'] <= terminal['elapsed_ns'],
                'completed command and original log identity')
    require(sum(row['elapsed_ns'] for row in receipt['commands']) <= terminal['elapsed_ns'],
            'command durations nested inside original phase')
    child = inventory(study)
    expected_members = {'registration-01.json', 'engineering-01.receipt.json', 'native-01.launch.json',
                        'native-01.terminal.json', 'native-01.log', 'engineering-01/command-0.log',
                        'engineering-01/command-1.log', 'source-snapshot-01/manifest.json'}
    expected_members.update('source-snapshot-01/' + name for name in plan['sources'])
    require(set(child) == expected_members, 'entire exact original88-file study inventory')
    # Interpret only the authenticated logs; no numerical source is imported.
    require((phase / 'command-0.log').read_text() == 'All checks passed!\n', 'original lint passed')
    test_log = (phase / 'command-1.log').read_text()
    match = re.fullmatch(r'\.{44}\s+\[100%\]\n44 passed in ([0-9]+(?:\.[0-9]+)?)s\n', test_log)
    require(match is not None, 'exact original 44-pass test log without hidden failures or skips')
    inputs = {str(study / name): pin for name, pin in child.items()}
    inputs.update({str(ROOT / name): pin for name, pin in plan['sources'].items()})
    inputs.update(prior['inputs'])
    inputs[str(Path(__file__).resolve())] = descriptor(Path(__file__).resolve())
    return {'plan': plan, 'plan_path': plan_path, 'receipt': receipt, 'receipt_path': receipt_path,
            'launch_path': launch_path, 'terminal': terminal, 'terminal_path': terminal_path,
            'child_files': child, 'prior': prior, 'inputs': inputs, 'pytest_seconds': float(match.group(1))}


def summarize(auth):
    return {
        'version': VERSION, 'study_version': STUDY_VERSION, 'status': 'QUALIFICATION_PASS',
        'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'worker_receipt': {'path': str(auth['receipt_path']), **descriptor(auth['receipt_path'])},
        'original_terminal': {'path': str(auth['terminal_path']), **descriptor(auth['terminal_path'])},
        'tests_passed': 44, 'tests_failed': 0, 'lint_passed': True, 'source_files': 80,
        'snapshot_source_files': 80, 'study_files': len(auth['child_files']),
        'original_wall_seconds': auth['terminal']['wall_seconds'],
        'original_timing_scope': auth['terminal']['timing_scope'],
        'pytest_reported_seconds': auth['pytest_seconds'],
        'commands': auth['receipt']['commands'], 'runtime': auth['plan']['runtime'],
        'prior_stop': auth['prior']['summary'],
        'prototype': {
            'module': 'openjev.research.finite_joint_reuse',
            'function': 'joint_objective(model, prefix, lengths, endpoint_positions, actions, observations, targets, *, total_attempts, total_survivors, total_events)',
            'arms': ['original_free', 'matched_free', 'rounded'], 'engineering_seed': 943101,
            'horizons': [1, 2, 8], 'value_absolute_and_relative_tolerance': 1e-10,
            'gradient_clipped_gradient_and_three_Adam_step_tolerance': 1e-9,
            'objective': 'n/b * (e/S * endpoint_loss + prefix_NLL_sum/E)',
            'checked_structural_work_for_nonempty_endpoint_batch': {
                'probability_field_builds': {'separate_routes': 5, 'reuse': 1},
                'eligible_endpoint_prefix_filters': {'separate_routes': 2, 'reuse': 1},
                'all_attempt_prefix_likelihood_filters': {'separate_routes': 1, 'reuse': 1},
            },
            'work_blocks': ['shared', 'endpoint_prefix', 'prefix_nll', 'blind', 'observed'],
            'autograd': 'Per-call shared graph; no persistent cache or detachment.',
            'reset_arithmetic': 'All-attempt NLL remains separate to preserve its sum/division order.',
            'readout_arithmetic': 'Per-horizon readouts remain separate for the two continuation routes.',
        },
        'scope': 'Fabricated joint-computation equivalence qualification only.',
        'limits': [
            'Structural counts are checked operations, not measured speedups or complete backward FLOPs.',
            'Numerical agreement does not establish bitwise gradients or identical long training trajectories.',
            'No empirical fit, task performance, architecture gain, scientific advancement or timing feasibility is admitted.',
            'The failed equal-update exposure prerequisite remains closed with its original threshold and counts.',
            'A fresh integration, independently audited saved outputs and bounded throughput comparison are required next.',
            'Timings describe this original qualification, not comparative model speed.',
            'The archive preserves source and evidence; interpreter and installed packages remain external.',
        ],
        'training_advances': False, 'scientific_execution_admitted': False,
        'publication_counts': PUBLICATION_COUNTS,
    }


def overview(summary):
    return (
        '# Joint computation reuse: component qualification\n\n'
        f'**QUALIFICATION_PASS: {summary["tests_passed"]} fabricated tests passed.** '
        'This establishes the checked numerical equivalence of a per-call computation-sharing helper. '
        'It is not a speed benchmark, empirical fit or scientific advancement.\n\n'
        f'The original native-supervised process completed in {summary["original_wall_seconds"]:.9f}s '
        f'under its 180-second cap. Pytest reported {summary["pytest_reported_seconds"]:.2f}s; '
        'that time is nested within the whole process. Lint also passed. '
        'All 80 registered sources and their original snapshot copies remain unchanged.\n\n'
        'The prototype API is `finite_joint_reuse.joint_objective(model, prefix, lengths, '
        'endpoint_positions, actions, observations, targets, *, total_attempts, total_survivors, '
        'total_events)`. It preserves the global reduction '
        '`n/b * (e/S * endpoint_loss + prefix_NLL_sum/E)`, with the same three model arms. '
        'Tests use fabricated public tokens and fixed targets at horizons 1, 2 and 8, including '
        'terminal/padded histories, empty endpoint batches and partial batches. No saved weights '
        'or scientific data are loaded by the qualification.\n\n'
        '| Checked work for a nonempty endpoint batch | Separate routes | Reuse |\n'
        '|---|---:|---:|\n'
        '| Probability-field construction | 5 | 1 |\n'
        '| Eligible endpoint-prefix filtering | 2 | 1 |\n'
        '| All-attempt prefix likelihood | 1 | 1 |\n\n'
        'Shared tensors stay attached to autograd and live only within a call. '
        'The all-attempt prefix likelihood retains its own reset arithmetic, and both continuations '
        'retain separate per-horizon readouts and prediction before observation assimilation. '
        'These operation counts do not measure speed or complete backward work.\n\n'
        'Values and losses are checked at absolute/relative tolerance 1e-10; raw gradients, '
        'clipped gradients and three Adam steps at 1e-9. Graph sharing can change floating-point '
        'accumulation order. The checks do not imply bitwise equivalence or identical longer training.\n\n'
        'The [previous equal-update feasibility stop](finite-update-learning-stop-results.md) '
        'remains failed and closed. This qualification does not change its 90-second projection '
        'threshold or registered update counts. A fresh integration, independently audited saved '
        'outputs and a bounded throughput comparison are still needed before a speed or feasibility claim.\n\n'
        '[Frozen protocol](finite-joint-reuse-qualification-protocol.md) · '
        '[Summary](finite-joint-reuse-qualification-results/summary.json) · '
        '[Manifest](finite-joint-reuse-qualification-results/manifest.json) · '
        '[Complete evidence archive](finite-joint-reuse-qualification-results/evidence.tar.gz) · '
        '[Publication receipt](finite-joint-reuse-qualification-results/receipt.json)\n\n'
        'The archive includes the complete child qualification, all 80 current source files and '
        'their snapshot copies, plus the explicitly bound prior-stop evidence and publication. '
        'Historical array/checkpoint artifacts are copied as opaque bytes only. '
        'The publisher performs zero array decodes, model calls or numerical replays. '
        'Installed packages and the interpreter remain external dependencies.\n'
    )


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


def publish(study, output, registration_sha256):
    study, output = Path(study).absolute(), Path(output).absolute()
    overview_path = ROOT / 'research/finite-joint-reuse-qualification-results.md'
    require(output == ROOT / 'research/finite-joint-reuse-qualification-results'
            and output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink(),
            'exclusive prescribed publication folder')
    require(not overview_path.exists() and not overview_path.is_symlink(), 'exclusive receipt-bound overview')
    auth = authenticate(study, registration_sha256)
    summary = summarize(auth)
    summary_payload = json_bytes(summary)
    overview_payload = overview(summary).encode()
    payloads = {}
    for name, pin in auth['inputs'].items():
        path = regular(Path(name))
        payload = path.read_bytes()
        require(byte_descriptor(payload) == pin, 'admitted evidence unchanged while assembling archive')
        payloads['repository/' + path.relative_to(ROOT).as_posix()] = payload
    source = Path(__file__).resolve()
    payloads['publication/summary.json'] = summary_payload
    payloads['publication/overview.md'] = overview_payload
    manifest = {
        'version': VERSION, 'registration': summary['registration'], 'scope': summary['scope'],
        'training_advances': False, 'sources': auth['plan']['sources'], 'inputs': auth['inputs'],
        'files': {name: byte_descriptor(payload) for name, payload in sorted(payloads.items())},
        'self_exclusion': 'files covers every other archive member; MANIFEST.json excludes itself.',
        'external_dependencies': {'runtime': auth['plan']['runtime'],
                                  'absolute_repository_root': str(ROOT), 'portable_execution_claimed': False},
    }
    manifest_payload = json_bytes(manifest)
    payloads['MANIFEST.json'] = manifest_payload
    output.mkdir()
    write(output / 'summary.json', summary_payload)
    write(overview_path, overview_payload)
    write(output / 'manifest.json', manifest_payload)
    roundtrip = archive(output / 'evidence.tar.gz', payloads)
    after = authenticate(study, registration_sha256)
    require(after == auth, 'all original input hashes, runtime, source pins and closures unchanged after publication')
    receipt = {
        'version': VERSION, 'status': 'PASS', 'registration': summary['registration'],
        'publisher': {'path': str(source), **descriptor(source)}, 'output': str(output),
        'files': {name: descriptor(output / name) for name in ('summary.json', 'manifest.json', 'evidence.tar.gz')},
        'overview': {'path': str(overview_path), **descriptor(overview_path)},
        'inputs_before': auth['inputs'], 'inputs_after': after['inputs'], 'inputs_unchanged': True,
        'archive_roundtrip': roundtrip, 'publication_counts': PUBLICATION_COUNTS,
        'training_advances': False, 'scientific_execution_admitted': False,
    }
    write(output / 'receipt.json', json_bytes(receipt))
    return {'output': str(output), 'receipt': descriptor(output / 'receipt.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
