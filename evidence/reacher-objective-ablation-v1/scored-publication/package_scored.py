from pathlib import Path
import gzip,hashlib,json,tarfile,time
root=Path(__file__).resolve().parents[2]
out=root/'output/reacher-objective-ablation-v1/scored-publication-v1'
out.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
ex=root/'runs/reacher-objective-ablation-v1/execution';au=root/'evidence/reacher-objective-ablation-v1/audit'
p=root/'evidence/reacher-objective-ablation-v1/protocol/plan.json'
assert sha(p)=='dca778153230375a5a0e790390d328d69857dd4c2d3260b61fb9f214fa1d32f2'
assert sha(au/'receipt.json')=='ee5f68ba9c194d39695c16d257bb05e3f294488a14c025e50a9e881ce4280dc8'
a=read(au/'receipt.json');c=read(ex/'completed.json');plan=read(p)
assert a['status']==c['status']=='completed' and a['engineering'] is False
assert not(ex/'failed.json').exists() and not(au/'failed.json').exists()
assert sha(ex/'completed.json')==a['execution_completed_sha256']
assert set(c['files'])=={str(x.relative_to(ex)) for x in ex.rglob('*') if x.is_file() and x.name!='completed.json'}
paths={p,ex/'completed.json',au/'receipt.json'}
for folder,members in [(ex,c['files']),(au,a['files']),(root,a['source_sha256'])]:
 for name,digest in members.items():
  q=folder/name;assert q.is_file() and not q.is_symlink() and sha(q)==digest;paths.add(q)
for name in ['LICENSE','pyproject.toml','uv.lock','evidence/reacher-objective-ablation-v1/launch.json','evidence/reacher-objective-ablation-v1/audit-launch.json','evidence/reacher-objective-ablation-v1/audit-terminal.json','runs/reacher-objective-ablation-v1/execution.log','runs/reacher-objective-ablation-v1/audit-launcher/audit.log']:
 paths.add(root/name)
rows=[{'path':str(q.relative_to(root)),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(paths)]
write(out/'manifest.json',{'scope':'Complete current execution, independent saved-output audit, frozen sources and process records. Historical lineage artifacts remain separate release dependencies.','members':rows,'files':len(rows),'bytes':sum(x['bytes'] for x in rows)})
archive=out/'scored-execution-and-audit.tar.gz'
with archive.open('xb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=1) as gz:
  with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
   for row in rows:
    q=root/row['path'];assert sha(q)==row['sha256']
    info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
    with q.open('rb') as f:tar.addfile(info,f)
print(json.dumps({'stage':'archive_written','bytes':archive.stat().st_size}),flush=True)
expected={x['path']:x for x in rows};seen=set()
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  assert member.isfile() and member.name in expected and member.name not in seen
  row=expected[member.name];assert member.size==row['bytes']
  with tar.extractfile(member) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
  seen.add(member.name)
assert seen==set(expected)
parts=[]
if archive.stat().st_size>1_900_000_000:
 with archive.open('rb') as inp:
  i=0
  while True:
   chunk=inp.read(900*1024*1024)
   if not chunk:break
   part=out/f'scored-execution-and-audit.tar.gz.part{i:02}'
   with part.open('xb') as f:f.write(chunk)
   parts.append({'name':part.name,'bytes':part.stat().st_size,'sha256':sha(part)});i+=1
else:parts=[{'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}]
write(out/'receipt.json',{'status':'verified','files':len(rows),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'manifest_sha256':sha(out/'manifest.json'),'parts':parts,'source_sha256':sha(Path(__file__)),'verified_unix':time.time(),'scientific_status':'Recorded gate remains failed; verification does not change it.'})
print(json.dumps({'status':'verified','archive_bytes':archive.stat().st_size,'parts':parts}),flush=True)
