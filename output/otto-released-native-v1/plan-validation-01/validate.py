"""One metadata-only check of the prospectively prepared native qualification plan."""
import hashlib
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PLAN = ROOT / 'output/otto-released-native-v1/qualification-plan-01.json'
PLAN_SHA = 'ee02c99cea3968a36498ae1ef8f3eefff4c92d39e6e3c1848e83666e7754bfd0'
SOURCE = ROOT / 'scripts/qualify_otto_released_native.py'
SOURCE_SHA = '96c68e94498c6b83eb2983c1183c444c5d1080eee06ff961562aeb56a845c65e'
HEAVY = ('numpy', 'scipy', 'tensorflow', 'tf_keras', 'h5py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def deadline(_signum, _frame):
    raise TimeoutError('Metadata authentication exceeded 60 seconds')


started = time.perf_counter()
record = {
    'scope': 'Metadata authentication only: raw byte hashes, saved JSON, package metadata. No tensor decoding or scientific execution.',
    'plan_sha256': PLAN_SHA, 'runner_sha256': SOURCE_SHA,
    'validator_sha256': sha(Path(__file__)), 'python_executable': sys.executable,
    'numerical_imports_before': [name for name in HEAVY if name in sys.modules],
    'native_calls': 0, 'model_calls': 0, 'tensor_array_decodes': 0,
}
write('started.json', record)
signal.signal(signal.SIGALRM, deadline)
signal.alarm(60)
try:
    assert sha(PLAN) == PLAN_SHA and sha(SOURCE) == SOURCE_SHA
    assert not record['numerical_imports_before']
    spec = importlib.util.spec_from_file_location('native_plan_authentication_only', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plan, paths = module.authenticate(SimpleNamespace(plan=PLAN, plan_sha256=PLAN_SHA))
    assert not any(name in sys.modules for name in HEAVY)
    assert sha(PLAN) == PLAN_SHA and sha(SOURCE) == SOURCE_SHA
    record.update(status='completed', agreement=True, source_files=len(plan['sources']),
                  runtime_distributions=len(plan['runtime']['all_distributions']),
                  input_roles=sorted(paths), numerical_imports_after=[],
                  elapsed_seconds=time.perf_counter() - started)
    signal.alarm(0)
    write('receipt.json', record)
    print(json.dumps({'status': 'completed', 'agreement': True,
                      'receipt_sha256': sha(OUT / 'receipt.json'),
                      'elapsed_seconds': record['elapsed_seconds']}))
except BaseException as error:
    signal.alarm(0)
    record.update(status='failed', agreement=False, error=repr(error),
                  traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
    write('failed.json', record)
    raise
