"""Recorded fabricated-only qualification, preserving every bounded attempt."""
import argparse, hashlib, json, os, signal, subprocess, time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--configuration',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=Path.cwd();config=json.loads(args.configuration.read_text());out=args.output
out.mkdir(exist_ok=False)
def descriptor(p):
 b=p.read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def pins():return {p:descriptor(root/p)['sha256'] for p in config['sources']}
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
save(out/'configuration.json',config)
(out/'runner.py').write_bytes(Path(__file__).read_bytes())
r={'status':'started','scope':'Fabricated engineering qualification only; no teacher/native/empirical work.', 'sources_before':pins(),'results':[], 'started_unix_ns':time.time_ns()}
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:env[name]='1'
try:
 if r['sources_before'] != config['sources']:raise RuntimeError('Source pin mismatch before execution')
 for i,cmd in enumerate(config['commands']):
  log=f'command-{i}.log';started=time.time_ns();tick=time.monotonic();timed=False
  with (out/log).open('xb') as stream:
   child=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=config['command_timeout_seconds'])
   except subprocess.TimeoutExpired:
    timed=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
  try:os.killpg(child.pid,0);absent=False
  except ProcessLookupError:absent=True
  item={'command':cmd,'exit_code':code,'log':log,'pid':child.pid,'elapsed_seconds':time.monotonic()-tick,'started_unix_ns':started,'finished_unix_ns':time.time_ns(),'reaped':child.poll() is not None,'group_absent':absent,'timed_out':timed}
  r['results'].append(item)
  print(json.dumps(item),flush=True)
  print((out/log).read_text()[-5000:],flush=True)
  if code!=0 or timed or not absent:raise RuntimeError('bounded qualification command did not pass')
 r['status']='passed'
except BaseException as error:
 r.update(status='failed',error=f'{type(error).__name__}: {error}')
finally:
 r['sources_after']=pins()
 if r['sources_after']!=r['sources_before']:r.update(status='failed',error='Source changed during qualification')
 r['finished_unix_ns']=time.time_ns();r['files']={p.name:descriptor(p) for p in sorted(out.iterdir()) if p.is_file()}
 save(out/'receipt.json',r)
 print(json.dumps({'status':r['status'],'receipt':str(out/'receipt.json'),**descriptor(out/'receipt.json')}),flush=True)
raise SystemExit(0 if r['status']=='passed' else 1)
