"""Publish one closed fabricated-input qualification using opaque source bytes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-rounded-transition-qualification-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    require(path.is_file() and not path.is_symlink(), 'ordinary input file')
    payload = path.read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def publish(study, registration_sha, destination):
    study, destination = study.resolve(), destination.resolve()
    publisher_pin = pin(Path(__file__).resolve())
    inputs = {}

    def read(name):
        path = study / name
        inputs[name] = pin(path)
        return json.loads(path.read_text())

    plan = read('registration-01.json')
    require(inputs['registration-01.json']['sha256'] == registration_sha, 'exact registration')
    require(plan['version'] == VERSION and plan['root'] == str(ROOT), 'study and root')
    receipt = read('engineering-01.receipt.json')
    launch, terminal = read('native-01.launch.json'), read('native-01.terminal.json')
    inputs['native-01.log'] = pin(study / 'native-01.log')
    require(receipt['status'] == 'PASS' and receipt['version'] == VERSION
            and receipt['plan'] == str(study / 'registration-01.json')
            and receipt['plan_sha256'] == registration_sha
            and receipt['output'] == str(study / 'engineering-01')
            and receipt['supervision'] == str(study / 'native-01.launch.json'), 'qualified original worker')
    require(receipt['launch'] == launch and receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
            'original source and launch identities')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['group_absent'] and not terminal['timed_out'] and terminal['timing_available']
            and terminal['cleanup']['reaped'] and not terminal['cleanup']['errors']
            and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['started_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'], 'closed original process')
    require(all(terminal[key] == launch[key] for key in ('command', 'cwd', 'pid', 'pgid', 'cap_seconds',
            'clock_backend', 'started_ns', 'deadline_ns', 'watchdog_sha256', 'clock_source_sha256')), 'joined terminal')
    require(terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + 90 * 10**9
            and terminal['wall_seconds'] == terminal['elapsed_ns'] / 1e9, 'native elapsed and deadline identities')
    expected = [plan['runtime']['executable'], str(ROOT / 'scripts/qualify_finite_rounded_transition.py'),
                'worker', '--plan', receipt['plan'], '--plan-sha256', registration_sha,
                '--supervision', receipt['supervision'], '--output', receipt['output']]
    require(launch['command'] == expected and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == 90,
            'exact worker invocation and cap')
    require(launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and launch['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'native implementation binding')
    require(plan['commands'] == [[str(ROOT / '.venv/bin/ruff'), 'check',
            'src/openjev/research/finite_rounded_transition.py', 'tests/test_finite_rounded_transition.py',
            'scripts/qualify_finite_rounded_transition.py'], [plan['runtime']['executable'], '-m', 'pytest',
            '-q', '-p', 'no:cacheprovider', '--noconftest', 'tests/test_finite_rounded_transition.py']], 'selected commands')
    require(len(receipt['commands']) == 2 and [row['command'] for row in receipt['commands']] == plan['commands']
            and all(row['returncode'] == 0 for row in receipt['commands']), 'both commands passed')
    require(set(receipt['files']) == {'command-0.log', 'command-1.log'}, 'exact logs')
    for i, row in enumerate(receipt['commands']):
        name = f'engineering-01/command-{i}.log'
        inputs[name] = pin(study / name)
        require(inputs[name] == row['log'] == receipt['files'][f'command-{i}.log'], 'original log bytes')
    require({p.name for p in (study / 'engineering-01').iterdir()} == set(receipt['files']), 'no extra phase files')
    manifest = read('source-snapshot-01/manifest.json')
    require(manifest['sources'] == plan['sources'] and manifest['registration'] == {
        'path': receipt['plan'], **inputs['registration-01.json']}, 'full original snapshot manifest')
    for name, expected_pin in plan['sources'].items():
        saved = 'source-snapshot-01/' + name
        inputs[saved] = pin(study / saved)
        require(inputs[saved] == expected_pin == pin(ROOT / name), 'unchanged original source: ' + name)
    require({str(p.relative_to(study / 'source-snapshot-01')) for p in (study / 'source-snapshot-01').rglob('*')
             if p.is_file()} == {*plan['sources'], 'manifest.json'}, 'complete exclusive source snapshot')
    log = (study / 'engineering-01/command-1.log').read_text()
    matches = re.findall(r'(?m)^(\d+) passed(?:, \d+ warnings?)? in ([0-9.]+)s$', log)
    require(len(matches) == 1 and 'FAILED' not in log, 'successful preserved pytest summary')
    count = int(matches[0][0])
    require(count > 0, 'nonempty numerical qualification')
    require(not destination.exists(), 'exclusive public output')
    destination.mkdir(parents=True)
    summary = {'version': VERSION, 'status': 'NUMERICAL_QUALIFICATION_PASS', 'tests_passed': count,
        'native_seconds': terminal['elapsed_ns'] / 1e9, 'registered_sources': len(plan['sources']),
        'registration_sha256': registration_sha, 'sweeps': 4, 'slack': 1e-8, 'tolerance': 1e-12,
        'training_calls': 0, 'model_calls': 0, 'empirical_evaluation_calls': 0,
        'scope': 'Fabricated forward, derivative and failure cases only. No learning or architecture claim.',
        'rss_scope': 'Peak recorded by original worker; no continuous memory-limit enforcement claimed.',
        'publisher_array_decodes': 0, 'publisher_numerical_calls': 0}
    write(destination / 'summary.json', summary)
    write(destination / 'manifest.json', {'study': str(study), 'inputs': inputs, 'publisher': publisher_pin,
                                         'current_source_pins': plan['sources']})
    archive = destination / 'evidence.tar.gz'
    payloads = {name: (study / name).read_bytes() for name in inputs}
    for name, payload in payloads.items():
        require({'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)} == inputs[name],
                'archive bytes match admitted descriptor')
    with archive.open('xb') as stream, \
            gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed, \
            tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name, payload in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(payloads) and {m.name for m in members} == set(payloads), 'exact archive roster')
        for member in members:
            require(member.isfile(), 'ordinary archive member')
            payload = tar.extractfile(member).read()
            require(payload == payloads[member.name]
                    and {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)} == inputs[member.name],
                    'opaque archive round trip matches original descriptor')
    require(all(pin(study / name) == descriptor for name, descriptor in inputs.items()), 'original evidence unchanged')
    require(all(pin(ROOT / name) == descriptor for name, descriptor in plan['sources'].items()),
            'current frozen sources unchanged after publication')
    require(pin(Path(__file__).resolve()) == publisher_pin, 'publisher unchanged')
    receipt_out = {'status': 'PASS', 'inputs': inputs, 'outputs': {p.name: pin(p) for p in destination.iterdir()},
                   'archive_members': len(payloads), 'archive_roundtrip': True, 'numerical_calls': 0, 'array_decodes': 0}
    write(destination / 'receipt.json', receipt_out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    result = publish(ROOT / 'output' / VERSION, args.registration_sha256,
                     ROOT / 'research/finite-rounded-transition-qualification-results')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
