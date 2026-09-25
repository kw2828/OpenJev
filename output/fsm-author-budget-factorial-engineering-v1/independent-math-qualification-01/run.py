import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
cfg=json.loads((out/'definition.json').read_text())
def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checks():
 for name,value in cfg['sources'].items():
  assert pin(root/name)==value,name
  snap=pin(out/'source'/name)
  assert all(snap[k]==value[k] for k in ('sha256','bytes')),name
checks();start=time.perf_counter();rows=[];status='PASS';error=None
try:
 for i,cmd in enumerate(cfg['commands'],1):
  path=out/f'command-{i:02}.log';before=time.perf_counter()
  with path.open('xb') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**cfg['environment']})
   try:code=process.wait(timeout=cfg['per_command_timeout_seconds'])
   except subprocess.TimeoutExpired:
    process.kill();code=process.wait();status='TIMEOUT'
  row={'command':cmd,'returncode':code,'seconds':time.perf_counter()-before,'log':pin(path),'pid':process.pid}
  rows.append(row);(out/f'command-{i:02}.json').write_text(json.dumps(row,indent=2)+'\n')
  print(path.read_text(),flush=True)
  if code!=0:
   status='FAIL' if status=='PASS' else status;break
 checks()
except BaseException as exc:
 status='FAIL';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
result={'status':status,'error':error,'seconds':time.perf_counter()-start,'commands':rows,'definition':pin(out/'definition.json'),'sources_before':cfg['sources'],'sources_after':{name:pin(root/name) for name in cfg['sources']},'scope':cfg['scope'],'wrapper':pin(Path(__file__).resolve())}
(out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':status,'seconds':result['seconds'],'receipt':pin(out/'receipt.json')}),flush=True)
raise SystemExit(0 if status=='PASS' else 1)
