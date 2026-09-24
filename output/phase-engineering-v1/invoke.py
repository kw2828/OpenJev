"""Exclusive original-process receipt wrapper; preserves failures, never retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--timeout',type=int,default=2400)
p.add_argument('argv',nargs=argparse.REMAINDER)
a=p.parse_args()
argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
root=Path.cwd()
out=root/'output/phase-engineering-v1'
paths=['src/openjev/research/phase_memory.py','src/openjev/research/silverbox_data.py',
       'scripts/phase_study.py','scripts/audit_phase_study.py',
       'tests/test_phase_memory.py','tests/test_silverbox_data.py',
       'tests/test_phase_study.py','tests/test_audit_phase_study.py','research/phase-protocol.md']
def pin(b):
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def pins():
    return {s:pin((root/s).read_bytes()) for s in paths}
threads={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}
env=dict(os.environ,**threads,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
         PYTEST_ADDOPTS='',PYTEST_PLUGINS='')
assert argv and not (out/(a.name+'.json')).exists()
assert not (out/(a.name+'.log')).exists()
before=pins()
started=time.monotonic()
status='EXITED'
with (out/(a.name+'.log')).open('xb') as stream:
    try:
        completed=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=a.timeout)
        code=completed.returncode
    except subprocess.TimeoutExpired:
        code=-9
        status='TIMEOUT'
    except Exception as error:
        stream.write((type(error).__name__+': '+str(error)).encode())
        code=-1
        status='START_FAILED'
receipt={'state':status,'returncode':code,'argv':argv,'elapsed_seconds':time.monotonic()-started,
         'thread_env':threads,'sources_before':before,'sources_after':pins(),
         'log_path':a.name+'.log','log':pin((out/(a.name+'.log')).read_bytes()),
         'timeout_seconds':a.timeout,'wrapper':pin(Path(__file__).read_bytes())}
with (out/(a.name+'.json')).open('x') as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps({k:receipt[k] for k in ('state','returncode','elapsed_seconds','log_path')}))
raise SystemExit(0 if code==0 else 1)
