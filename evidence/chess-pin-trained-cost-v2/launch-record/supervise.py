from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
plan=root/'evidence/chess-pin-trained-cost-v2/protocol/plan.json'
expected='46bd0965241aa8fa8e0a54c5f19a925dfd6f5102a93b97eefc8abe4832abe464'
def write(name,data):
 with (out/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
write('supervisor-started.json',{'pid':os.getpid(),'unix':time.time(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
for phase in ('run','audit'):
 cmd=[str(root/'.venv/bin/python'),str(root/'scripts/chess_pin_trained_cost_recovery.py'),phase,'--plan',str(plan),'--out',str(root/('runs/chess-pin-trained-cost-v2/execution' if phase=='run' else 'evidence/chess-pin-trained-cost-v2/audit'))]
 if phase=='audit':cmd+=['--execution',str(root/'runs/chess-pin-trained-cost-v2/execution')]
 start=time.monotonic(); begun=time.time(); error=None; timed_out=False; rc=None
 with (out/f'{phase}.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  write(f'{phase}-started.json',{'command':cmd,'pid':p.pid,'unix':begun,'plan_sha256':expected})
  try:rc=p.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   timed_out=True;p.kill();rc=p.wait();error='Frozen1800s outer phase cap exceeded; no retry'
 elapsed=time.monotonic()-start
 write(f'{phase}-terminal.json',{'status':'timed_out' if timed_out else 'exited','returncode':rc,'wall_seconds':elapsed,'started_unix':begun,'finished_unix':time.time(),'error':error,'plan_sha256':expected})
 if rc!=0:
  write('terminal.json',{'status':'failed','phase':phase,'returncode':rc,'retry':False,'error':error});raise SystemExit(1)
 if phase=='run':
  execution=root/'runs/chess-pin-trained-cost-v2/execution'
  receipt=json.loads((execution/'completed.json').read_text())
  assert receipt['status']=='completed' and receipt['plan_sha256']==expected and not (execution/'failed.json').exists()
  assert set(receipt['files'])=={str(x.relative_to(execution)) for x in execution.rglob('*') if x.is_file() and x.name!='completed.json'}
  for name,digest in receipt['files'].items():assert hashlib.sha256((execution/name).read_bytes()).hexdigest()==digest
 else:
  audit=root/'evidence/chess-pin-trained-cost-v2/audit'
  receipt=json.loads((audit/'receipt.json').read_text())
  assert receipt['status']=='completed' and receipt['plan_sha256']==expected and not (audit/'failed.json').exists()
write('terminal.json',{'status':'completed','retry':False,'unix':time.time(),'plan_sha256':expected,'audit_receipt_sha256':hashlib.sha256((root/'evidence/chess-pin-trained-cost-v2/audit/receipt.json').read_bytes()).hexdigest()})
