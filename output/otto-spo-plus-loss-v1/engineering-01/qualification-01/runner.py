"""Record bounded fabricated-only checks without opening empirical arrays."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, out = Path.cwd(), args.output
configuration = json.loads(args.configuration.read_text())
out.mkdir(exist_ok=False)


def pin(path):
    content = path.read_bytes()
    return {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def sources():
    return {name: pin(root / name) for name in configuration['sources']}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


save(out / 'configuration.json', configuration)
with (out / 'runner.py').open('xb') as stream:
    stream.write(Path(__file__).read_bytes())
receipt = {'status': 'started', 'all_passed': False, 'empirical_array_reads': 0,
           'model_calls': 0, 'native_calls': 0, 'optimizer_calls': 0,
           'scope': 'Fabricated engineering checks only.', 'sources': sources(),
           'started_unix_ns': time.time_ns(), 'results': []}
environment = os.environ.copy()
environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='src')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    environment[name] = '1'
try:
    if receipt['sources'] != configuration['sources']:
        raise RuntimeError('Sources changed before qualification')
    for index, command in enumerate(configuration['commands']):
        log = out / f'command-{index}.log'
        started, tick, timed_out = time.time_ns(), time.monotonic(), False
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=root, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=configuration['command_timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        try:
            os.killpg(child.pid, 0)
            group_absent = False
        except ProcessLookupError:
            group_absent = True
        result = {'command': command, 'exit_code': code, 'log': log.name,
                  'pid': child.pid, 'reaped': child.poll() is not None,
                  'group_absent': group_absent, 'timed_out': timed_out,
                  'started_unix_ns': started, 'finished_unix_ns': time.time_ns(),
                  'elapsed_seconds': time.monotonic() - tick}
        receipt['results'].append(result)
        print(json.dumps(result), flush=True)
        print(log.read_text()[-5000:], flush=True)
        if code != 0 or timed_out or not group_absent:
            raise RuntimeError('Qualification command failed')
    receipt.update(status='completed', all_passed=True)
except BaseException as error:
    receipt.update(status='failed', all_passed=False, error=repr(error))
finally:
    receipt['sources_after'] = sources()
    if receipt['sources_after'] != receipt['sources']:
        receipt.update(status='failed', all_passed=False, error='Sources changed during checks')
    receipt['finished_unix_ns'] = time.time_ns()
    receipt['files'] = {p.name: pin(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(out / 'receipt.json'),
                      **pin(out / 'receipt.json')}), flush=True)
raise SystemExit(0 if receipt['all_passed'] else 1)
