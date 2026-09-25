"""One retained fabricated-only plot qualification; no empirical admission."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent/'qualification-01'
SOURCES = ('scripts/plot_fsm_author_factorial.py', 'tests/test_plot_fsm_author_factorial.py',
           'scripts/audit_fsm_author_factorial.py', 'scripts/fsm_nllfr_budget_audit_math.py',
           'scripts/audit_fsm_author_nllfr.py')
EXPECTED = ('ab72bd7733f762ed8e1f49ce15b09afafa5df2bab77038f69fcbe0c35fd94220',
            '6856c4124041d2ff5b2f986e67a0ed6e1a625a4a94f47c90fbcd75118b6ab7c2')
ENV = {**dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                      'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'), '1'),
       'PYTHONDONTWRITEBYTECODE':'1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1', 'MPLBACKEND':'Agg'}
COMMANDS = [['.venv/bin/ruff','check',*SOURCES[:2]],
            ['.venv/bin/python','-m','pytest','--noconftest','-q',SOURCES[1],
             '--basetemp',str(OUT/'fabricated-tmp')]]

def pin(path):
    path=Path(path); data=path.read_bytes()
    return {'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def save(path,value):
    with path.open('x') as handle: handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    assert Path.cwd().resolve()==ROOT
    before={name:pin(ROOT/name) for name in SOURCES}
    assert tuple(before[n]['sha256'] for n in SOURCES[:2])==EXPECTED
    OUT.mkdir(exist_ok=False)
    for name in SOURCES:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    definition={'sources':before,'commands':COMMANDS,'environment':ENV,'timeout_seconds':180,
                'wrapper':pin(__file__),'scope':'Fabricated scalar/opaque files and temporary figure only. No empirical output reads.'}
    save(OUT/'definition.json',definition)
    receipt={'status':'FAIL','definition':pin(OUT/'definition.json'),'sources_before':before,
             'commands':[],'started_time_ns':time.time_ns()}
    start=time.monotonic()
    try:
        for i,command in enumerate(COMMANDS,1):
            log=OUT/f'command-{i:02d}.log';t=time.monotonic()
            with log.open('xb') as handle:
                try:
                    code=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENV},stdout=handle,
                                        stderr=subprocess.STDOUT,timeout=180,check=False).returncode
                    row={'command':command,'returncode':code}
                except subprocess.TimeoutExpired:
                    row={'command':command,'returncode':None,'error':'timeout'}
            receipt['commands'].append({**row,'seconds':time.monotonic()-t,'log':pin(log)})
            if row['returncode']!=0:break
        receipt['sources_after']={name:pin(ROOT/name) for name in SOURCES}
        receipt['sources_unchanged']=before==receipt['sources_after']
        if (len(receipt['commands'])==2 and all(r['returncode']==0 for r in receipt['commands'])
                and receipt['sources_unchanged']):receipt['status']='PASS'
    except BaseException as exc:
        receipt['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        receipt.update(finished_time_ns=time.time_ns(),seconds=time.monotonic()-start)
        save(OUT/'receipt.json',receipt)
        print(json.dumps({'status':receipt['status'],'receipt':pin(OUT/'receipt.json')},indent=2))
    raise SystemExit(0 if receipt['status']=='PASS' else 1)
