"""Original combined qualification after the preserved attempt-01 failure."""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'attempt-01'
RUSTC = Path('/Users/kevinwu/.cache/openjev-rust/1.98.1/bin/rustc')
PYTHON = ROOT / '.venv/bin/python'
THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                'NUMEXPR_NUM_THREADS')}
HELD = {
    'rust/robot_transition/src/lib.rs': 'fb458ebec322d9a551b2693aaa95c5c58c048bf9925c31b077f1fc51da5b858b',
    'rust/robot_transition/src/gru_reference.rs': 'e0cb356f250c60d6d7a7b38182e2151627dc39bb647b5d07522b5a9f6a542d6e',
    'src/openjev/research/native_robot_transition.py': '89a6093414d45762c2cd1c8644ba6f04be5350104c4ef053ceb83ad958ec1d03',
    'src/openjev/research/native_robot_gru.py': '4145c786c3df6f0e9093ee294397b2c91fa47f2dbd87e28630cfa0d8b0395d50',
    'tests/test_native_robot_transition.py': 'cd9b9d4d058773709b7015a4c6749131d1dadc96159e81f5489dd3c8953f13c1',
    'tests/test_native_robot_gru.py': 'b65ae9333cdd59cecd455d5d3f06e2bf669ad265aca0e80704b7edea34ace2cc',
    'scripts/build_robot_native.py': 'eff6325d3ac8f21a556b93a72017eea9e1d8ce2f2bfa51beef0a0c5e045f9a39',
}
ORACLE_SOURCES = (
    'src/openjev/research/structured_robot_transition.py',
    'src/openjev/research/compact_robot_gate.py',
    'scripts/robot_structured_study.py', 'scripts/robot_transition_study.py',
    'scripts/robot_coupling_study.py',
    'src/openjev/research/bounded_robot_transition.py',
    'src/openjev/research/causal_robot_ridge.py',
    'src/openjev/research/joint_coupling.py',
    'src/openjev/research/industrial_robot_data.py', 'pyproject.toml',
)


def descriptor(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def inventory(folder):
    return {str(p.relative_to(folder)): descriptor(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def execute(name, command, env):
    started = time.time()
    clock = time.perf_counter()
    log = OUT / (name + '.log')
    error, returncode, state = None, None, 'ERROR'
    with log.open('xb') as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=120)
                state = 'EXITED'
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode, state, error = process.returncode, 'TIMEOUT', str(exc)
        except BaseException as exc:
            error = repr(exc)
    receipt = {'command': command, 'state': state, 'returncode': returncode,
               'started_unix': started, 'finished_unix': time.time(),
               'elapsed_seconds': time.perf_counter() - clock,
               'timeout_seconds': 120, 'error': error, 'log': descriptor(log)}
    save(name + '-process.json', receipt)
    return receipt


def main():
    os.chdir(ROOT)
    names = tuple(HELD) + ORACLE_SOURCES
    before = {name: descriptor(ROOT / name) for name in names}
    assert all(before[name]['sha256'] == sha for name, sha in HELD.items())
    prior = inventory(PRIOR)
    assert prior['qualification.json']['sha256'] == '69ee45ce8cf3e70c83b68ae9a95a4ee91f83a2a70e5d71f09a8ed5e6ab1c5732'
    assert json.loads((PRIOR / 'qualification.json').read_text())['status'] == 'FAILED'
    diagnostic = inventory(OUT.parent / 'reduction-diagnostic-01')
    for name in names:
        target = OUT / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
    library = OUT / 'build/libopenjev_robot.dylib'
    lint_files = [name for name in HELD if name.endswith('.py')]
    lint = [str(PYTHON), '-m', 'ruff', 'check', *lint_files]
    build = [str(PYTHON), 'scripts/build_robot_native.py', '--rustc', str(RUSTC),
             '--output', str(OUT / 'build')]
    tests = [str(PYTHON), '-m', 'pytest', '--noconftest', '-q', '-rA',
             'tests/test_native_robot_transition.py', 'tests/test_native_robot_gru.py',
             '--junitxml=' + str(OUT / 'pytest.xml')]
    env = dict(os.environ, **THREADS, ROBOT_TRANSITION_LIBRARY=str(library))
    save('preflight.json', {
        'version': 'native-robot-combined-fabricated-qualification-v2',
        'scope': 'Four structured families plus GRU32 fabricated inference only; no measured checkpoints/data, training or performance benchmark.',
        'tolerance': {'rtol': 1e-5, 'atol': 1e-5}, 'horizons': [128, 512],
        'expected_tests': 110, 'required_skips': 0,
        'compiler': descriptor(RUSTC), 'python': descriptor(PYTHON),
        'platform': platform.platform(), 'sources_before': before,
        'prior_failed_attempt': prior, 'source_derived_reduction_diagnostic': diagnostic,
        'qualification_script': descriptor(__file__), 'thread_env': THREADS,
        'commands': {'lint': lint, 'build': build, 'tests': tests},
        'compiler_policy': 'opt-level=3; no fast-math; target CPU default; no target-cpu override',
        'created_unix': time.time(),
    })
    processes = {}
    for name, command in [('lint-01', lint), ('build-01', build), ('test-01', tests)]:
        if name == 'test-01':
            built = json.loads((OUT / 'build/receipt.json').read_text())
            assert built['status'] == 'PASS' and built['sources_unchanged'] is True
            for rust in ('lib.rs', 'gru_reference.rs'):
                assert descriptor(OUT / 'build/source' / rust)['sha256'] == HELD['rust/robot_transition/src/' + rust]
            save('test-admission.json', {'library': descriptor(library),
                'build_receipt': descriptor(OUT / 'build/receipt.json'),
                'preflight': descriptor(OUT / 'preflight.json'), 'created_unix': time.time()})
        result = execute(name, command, env)
        processes[name] = result
        if result['state'] != 'EXITED' or result['returncode'] != 0:
            break
    after = {name: descriptor(ROOT / name) for name in names}
    cases = ET.parse(OUT / 'pytest.xml').getroot().findall('.//testcase') if (OUT / 'pytest.xml').is_file() else []
    counts = {'tests': len(cases), 'failed': sum(c.find('failure') is not None for c in cases),
              'errors': sum(c.find('error') is not None for c in cases),
              'skipped': sum(c.find('skipped') is not None for c in cases)}
    intact = before == after and prior == inventory(PRIOR) and diagnostic == inventory(OUT.parent / 'reduction-diagnostic-01')
    passed = (len(processes) == 3 and all(p['state'] == 'EXITED' and p['returncode'] == 0 for p in processes.values())
              and counts == {'tests': 110, 'failed': 0, 'errors': 0, 'skipped': 0} and intact)
    save('qualification.json', {'status': 'PASS' if passed else 'FAILED', 'counts': counts,
        'sources_before': before, 'sources_after': after, 'evidence_unchanged': intact,
        'preflight': descriptor(OUT / 'preflight.json'),
        'process_receipts': {name: descriptor(OUT / (name + '-process.json')) for name in processes},
        'library': descriptor(library) if library.is_file() else None,
        'tolerance': {'rtol': 1e-5, 'atol': 1e-5}, 'finished_unix': time.time()})
    print(json.dumps({'status': 'PASS' if passed else 'FAILED', 'counts': counts,
        'qualification': descriptor(OUT / 'qualification.json')}, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
