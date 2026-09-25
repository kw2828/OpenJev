import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
NAMES=('scripts/plot_fsm_author_nllfr.py','tests/test_plot_fsm_author_nllfr.py','scripts/plot_fsm_author_bla.py','scripts/plot_fsm_linear_controls.py','scripts/audit_fsm_author_nllfr.py','tests/test_audit_fsm_author_nllfr.py')
ENV={**dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'),'1'),'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(OUT/'matplotlib-cache')}
COMMANDS=[['.venv/bin/ruff','check',*NAMES[:2]],['.venv/bin/python','-m','pytest','--noconftest','-q',NAMES[1]]]
def pin(p):
 b=p.read_bytes(); return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def save(n,x):
 (OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
before={n:pin(ROOT/n) for n in NAMES};snapshots={}
for n in NAMES:
 p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());snapshots[n]=pin(p)
definition={'sources':before,'snapshots':snapshots,'commands':COMMANDS,'environment':ENV,'scope':'Fabricated scalars and opaque files only; one temporary synthetic figure; no measured inputs/models.'}
save('definition.json',definition)
record={'state':'RUNNING','status':'RUNNING','sources_before':before,'definition':pin(OUT/'definition.json'),'commands':[]}
save('receipt.json',record);start=time.monotonic();rc=0
try:
 for i,command in enumerate(COMMANDS,1):
  began=time.monotonic();log=OUT/f'command-{i:02d}.log'
  with log.open('xb') as handle:
   result=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENV},stdout=handle,stderr=subprocess.STDOUT,timeout=180)
  record['commands'].append({'command':command,'returncode':result.returncode,'seconds':time.monotonic()-began,'log':pin(log)})
  save('receipt.json',record)
  if result.returncode:rc=result.returncode;break
except BaseException as exc:
 rc=1;record['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
 record['sources_after']={n:pin(ROOT/n) for n in NAMES}
 record['sources_unchanged']=before==record['sources_after']
 record['state']='EXITED';record['returncode']=rc;record['seconds']=time.monotonic()-start
 record['status']='PASS' if rc==0 and record['sources_unchanged'] and len(record['commands'])==2 else 'FAIL'
 save('receipt.json',record)
 print(json.dumps(record,indent=2));raise SystemExit(0 if record['status']=='PASS' else 1)
