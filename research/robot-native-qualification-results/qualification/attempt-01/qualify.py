"""One original fabricated build/qualification attempt; no scientific data."""
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUSTC = Path('/Users/kevinwu/.cache/openjev-rust/1.98.1/bin/rustc')
PYTHON = ROOT / '.venv/bin/python'
THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                'NUMEXPR_NUM_THREADS')}


def descriptor(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def execute(name, command, env, timeout):
    started = time.time()
    clock = time.perf_counter()
    log = OUT / (name + '.log')
    error = None
    with log.open('xb') as stream:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            returncode = completed.returncode
            state = 'EXITED'
        except subprocess.TimeoutExpired as exc:
            returncode, state, error = None, 'TIMEOUT', str(exc)
        except BaseException as exc:
            returncode, state, error = None, 'ERROR', repr(exc)
    receipt = {'command': command, 'state': state, 'returncode': returncode,
               'started_unix': started, 'finished_unix': time.time(),
               'elapsed_seconds': time.perf_counter() - clock,
               'timeout_seconds': timeout, 'error': error, 'log': descriptor(log)}
    save(name + '-process.json', receipt)
    return receipt


def main():
    os.chdir(ROOT)
    source_names = ['tests/test_native_robot_transition.py',
                    'src/openjev/research/native_robot_transition.py',
                    'src/openjev/research/structured_robot_transition.py',
                    'src/openjev/research/compact_robot_gate.py', 'pyproject.toml']
    before = {name: descriptor(ROOT / name) for name in source_names}
    snapshot = descriptor(OUT / 'source/lib.rs')
    assert snapshot['sha256'] == '03029598b650ba1302bb90939ed9d10783d6183cb1be4c440ef8ab78b72f579c'
    assert before[source_names[0]]['sha256'] == 'cd9b9d4d058773709b7015a4c6749131d1dadc96159e81f5489dd3c8953f13c1'
    for name in source_names:
        target = OUT / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    version = subprocess.run([str(RUSTC), '-vV'], check=True, capture_output=True, text=True).stdout
    library = OUT / 'librobot_transition.dylib'
    build_command = [str(RUSTC), '--edition=2021', '--crate-type=cdylib',
                     '-C', 'opt-level=3', str(OUT / 'source/lib.rs'), '-o', str(library)]
    test_command = [str(PYTHON), '-m', 'pytest', '--noconftest', '-q', '-rA',
                    'tests/test_native_robot_transition.py',
                    '--junitxml=' + str(OUT / 'pytest.xml')]
    env = dict(os.environ, **THREADS, ROBOT_TRANSITION_LIBRARY=str(library))
    preflight = {'version': 'native-robot-transition-fabricated-qualification-v1',
                 'scope': 'Four-family fabricated inference parity only; no training, measured inputs or performance benchmark.',
                 'tolerance': {'rtol': 1e-5, 'atol': 1e-5}, 'horizons': [128, 512],
                 'expected_tests': 90, 'required_skips': 0, 'seed': 97241,
                 'compiler': descriptor(RUSTC), 'compiler_version': version,
                 'platform': platform.platform(), 'python': descriptor(PYTHON),
                 'source_snapshot': snapshot, 'sources_before': before,
                 'qualification_script': descriptor(__file__),
                 'thread_env': THREADS, 'build_command': build_command,
                 'test_command': test_command,
                 'compiler_policy': 'opt-level=3; no fast-math; target CPU default; no target-cpu override',
                 'created_unix': time.time()}
    save('preflight.json', preflight)
    build = execute('build-01', build_command, env, 120)
    if build['state'] != 'EXITED' or build['returncode'] != 0:
        return 1
    save('test-admission.json', {'library': descriptor(library), 'preflight': descriptor(OUT / 'preflight.json'),
                                 'build_receipt': descriptor(OUT / 'build-01-process.json'),
                                 'created_unix': time.time()})
    test = execute('test-01', test_command, env, 120)
    after = {name: descriptor(ROOT / name) for name in source_names}
    cases = ET.parse(OUT / 'pytest.xml').getroot().findall('.//testcase') if (OUT / 'pytest.xml').is_file() else []
    counts = {'tests': len(cases), 'failed': sum(c.find('failure') is not None for c in cases),
              'errors': sum(c.find('error') is not None for c in cases),
              'skipped': sum(c.find('skipped') is not None for c in cases)}
    passed = (test['state'] == 'EXITED' and test['returncode'] == 0
              and counts == {'tests': 90, 'failed': 0, 'errors': 0, 'skipped': 0}
              and before == after and descriptor(OUT / 'source/lib.rs') == snapshot)
    final = {'status': 'PASS' if passed else 'FAILED', 'counts': counts,
             'sources_before': before, 'sources_after': after,
             'source_snapshot': snapshot, 'library': descriptor(library),
             'preflight': descriptor(OUT / 'preflight.json'),
             'build_receipt': descriptor(OUT / 'build-01-process.json'),
             'test_receipt': descriptor(OUT / 'test-01-process.json'),
             'test_admission': descriptor(OUT / 'test-admission.json'),
             'tolerance': {'rtol': 1e-5, 'atol': 1e-5}, 'finished_unix': time.time()}
    save('qualification.json', final)
    print(json.dumps({'status': final['status'], 'counts': counts,
                      'test_seconds': test['elapsed_seconds'],
                      'qualification': descriptor(OUT / 'qualification.json')}, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
