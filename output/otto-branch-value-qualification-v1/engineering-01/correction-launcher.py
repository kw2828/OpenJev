import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=root/'output/otto-branch-value-qualification-v1/engineering-01'
files=['src/openjev/research/otto_value_branches.py','tests/test_otto_value_branches.py']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):
    with (out/name).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
pins={p:sha(root/p) for p in files}
commands=[('pytest-affected',[str(root/'.venv/bin/python'),'-m','pytest','-q',files[1],'-k','scalar or invalid_action_metadata']),('ruff-final',[str(root/'.venv/bin/python'),'-m','ruff','check',*files])]
write('correction-01.json',{'reason':'Initial57 synthetic tests passed; Ruff then identified three mechanical style/type-contract issues. Only invalid eligible-container exceptionTypeError, import order, and Python max spelling changed.','prior_receipt_sha256':sha(out/'receipt.json'),'sources':pins,'commands':dict(commands),'numerical_production_logic_changed':False,'limits':{'seconds':60,'threads':1,'rss_bytes':2*1024**3}})
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
env['PYTHONPATH']=str(root/'src');start=time.monotonic();records=[]
for name,command in commands:
    t=time.monotonic()
    with (out/(name+'.log')).open('xb') as log:
        try:
            p=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,60-(time.monotonic()-start)),check=False)
            r={'name':name,'command':command,'exit_code':p.returncode,'timed_out':False}
        except subprocess.TimeoutExpired:r={'name':name,'command':command,'exit_code':None,'timed_out':True}
    r['wall_seconds']=time.monotonic()-t;records.append(r);print((out/(name+'.log')).read_text())
    if r['timed_out']:break
elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
unchanged=all(sha(root/p)==h for p,h in pins.items());ok=len(records)==2 and all(r['exit_code']==0 for r in records) and unchanged and elapsed<60 and rss<=2*1024**3
receipt={'status':'completed' if ok else 'failed','prior_receipt_sha256':sha(out/'receipt.json'),'correction_sha256':sha(out/'correction-01.json'),'sources':pins,'sources_unchanged':unchanged,'commands':records,'wall_seconds':elapsed,'peak_child_rss_bytes':rss,'scope':'Affected synthetic tests and lint after mechanical corrections only; no simulator, model, training, checkpoint, evaluation or API calls.','initial_source_tests_passed':57,'final_source_validation':'Affected tests only; unchanged numerical implementation retains initial passing evidence','files':{name:{'sha256':sha(out/name),'bytes':(out/name).stat().st_size} for name in ['pytest-affected.log','ruff-final.log','correction-01.json']}}
write('receipt-02.json',receipt)
print(json.dumps({'status':receipt['status'],'receipt_sha256':sha(out/'receipt-02.json'),'sources':pins,'wall_seconds':elapsed,'peak_child_rss_bytes':rss}))
raise SystemExit(0 if ok else 1)
