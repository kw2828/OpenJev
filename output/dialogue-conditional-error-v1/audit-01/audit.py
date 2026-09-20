"""Independent saved diagnostic audit; stdlib and NumPy only, no inference."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
OUT=Path(__file__).resolve().parent
DIAG=ROOT/'output/dialogue-conditional-error-v1/diagnostic-01'
RUN=ROOT/'runs/dialogue-conditional-v1/training-01'
PREP=ROOT/'runs/dialogue-conditional-v1/preparation-01'
ORIGINAL=ROOT/'output/dialogue-conditional-v1/analysis-01/summary.json'
TRAIN_AUDIT=ROOT/'output/dialogue-conditional-v1/training-audit-01'
COMMIT='f4942dbe2af8dc19949d5455304f72361645ba02'
SUMMARY_SHA='c696c3f60d394f026d3cedea262df782b382198151fd40e254c12626398a1689'
DIAG_RECEIPT_SHA='c769b6bf048cea8a70dbc6101ff3e1816d166965882c9a7ab25a943f7fc3ec87'
RUN_SHA='df3c172bae7163292b54cdd9a3b1d6e3bb5a07acb49b62be4f0c0dfc68efbe75'
ORIGINAL_SHA='fb065bcc05550734f7fbe8a24def71b4b69ae5288c06dabbe1caddf39fb50e20'
PREP_SHA='960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3'
PRIOR_SUMMARY_SHA='58407b424b6a8d00409b94913de001f67af6910f18e58a8bcf104fed65c6840c'
PRIOR_RECEIPT_SHA='c9ecf01993c847536b983ae42f2dc6a1144b7ed318ba810401a3cea7685599c8'
SOURCE_PINS={'scripts/diagnose_dialogue_conditional.py':'68429dfb01ef98020e53ec5884443967e64df4e7789bdc80b1239c9dc752fc0e','tests/test_diagnose_dialogue_conditional.py':'4cb6ff2e4d6dbe52bb1953a5c88f8e6ec0812a924941e0e0d8297af013ff0351','research/dialogue-conditional-error-protocol.md':'9ee166f8f212fa89d796b82b59e475bac491deac8f0dff7b6cfc6821c10b5690'}
VALUES=('none','true','false','dontcare','other')
BINS=('unmentioned_retention','assigned_retention','first_assignment','revision','clear')
SEEDS=(5301,5302,5303); MODES=('mean','slot','candidate')
FITS={f'{m}-{s}' for m in MODES for s in SEEDS}
INPUTS={}


def check(ok,label):
    if not ok: raise ValueError(label)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def decode(text):
    def pairs(items):
        result={}
        for k,v in items:
            check(k not in result,'Duplicate JSON key'); result[k]=v
        return result
    def invalid(value): raise ValueError('Nonfinite JSON scalar '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def read(p): return decode(Path(p).read_text())


def bind(p,pin,size=None):
    p=Path(p).resolve(); check(p.is_relative_to(ROOT) and p.is_file(),'Input boundary')
    name=p.relative_to(ROOT).as_posix()
    if name not in INPUTS: INPUTS[name]={'sha256':sha(p),'bytes':p.stat().st_size}
    got=INPUTS[name]; check(got['sha256']==pin and (size is None or got['bytes']==size),'Hash/bytes '+name)


def manifest(base,mapping,expected):
    check(set(mapping)==set(expected),'Manifest exact keys')
    check({p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(expected)|{'completed.json'},'Exact execution members')
    for name,item in mapping.items():
        path=base/name
        check(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(),'Unsafe payload')
        bind(path,item['sha256'],item['bytes'])


def compare(actual,expected,path):
    if type(expected) is dict:
        check(type(actual) is dict and set(actual)==set(expected),'Fields '+path)
        for key in expected: compare(actual[key],expected[key],path+'/'+str(key))
    elif type(expected) is list:
        check(type(actual) is list and len(actual)==len(expected),'List '+path)
        for i,(a,b) in enumerate(zip(actual,expected,strict=True)): compare(a,b,path+'/'+str(i))
    elif type(expected) is float:
        check(type(actual) in (float,int) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),'Arithmetic '+path)
    else: check(type(actual) is type(expected) and actual==expected,'Identity '+path)


def authenticate():
    for p,pin in [(DIAG/'summary.json',SUMMARY_SHA),(DIAG/'receipt.json',DIAG_RECEIPT_SHA),(RUN/'completed.json',RUN_SHA),(PREP/'completed.json',PREP_SHA),(ORIGINAL,ORIGINAL_SHA),(TRAIN_AUDIT/'summary.json',PRIOR_SUMMARY_SHA),(TRAIN_AUDIT/'receipt.json',PRIOR_RECEIPT_SHA)]: bind(p,pin)
    check({p.name for p in DIAG.iterdir() if p.is_file()}=={'summary.json','receipt.json'},'Diagnostic exact output closure')
    receipt=read(DIAG/'receipt.json'); original=read(ORIGINAL); prior=read(TRAIN_AUDIT/'summary.json'); pr=read(TRAIN_AUDIT/'receipt.json')
    check(receipt['status']=='completed' and receipt['summary_sha256']==SUMMARY_SHA and receipt['input_run_completed_sha256']==RUN_SHA and receipt['input_analysis_sha256']==ORIGINAL_SHA,'Diagnostic input/output pins')
    check(receipt['model_calls']==receipt['training_calls']==0 and receipt['posthoc'] is True and receipt['comparison_result_changed'] is False and receipt['original_continuation_allowed'] is False,'Recorded diagnostic scope')
    check(prior['status']=='completed' and prior['all_reported_metrics_and_decisions_agree'] is True and prior['execution_completed_sha256']==RUN_SHA and pr['files']['summary.json']['sha256']==PRIOR_SUMMARY_SHA,'Inherited independent training audit')
    check(original['continuation_allowed'] is False and prior['continuation_allowed'] is False,'Original rule remains failed')
    mapped={k:SOURCE_PINS[n] for k,n in zip(('source','tests','protocol'),SOURCE_PINS,strict=True)}
    compare(receipt['source_pins'],mapped,'diagnostic source pins')
    for name,pin in SOURCE_PINS.items():
        bind(ROOT/name,pin)
        raw=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
        check(hashlib.sha256(raw).hexdigest()==pin,'Preanalysis Git source '+name)
    pub=ROOT/'output/dialogue-conditional-error-v1/publication-preanalysis-01/receipt.json'
    bind(pub,'925dbda567ce4e17d6642eff0ce3f04002826da5e8d952201f8c32f6aaaa95f8')
    publication=read(pub); check(publication['commit']==publication['remote_main']==COMMIT and publication['files']==SOURCE_PINS,'Publication witness')
    check(subprocess.run(['git','merge-base','--is-ancestor',COMMIT,'HEAD'],cwd=ROOT,capture_output=True).returncode==0,'Local ancestry')
    done=read(RUN/'completed.json')
    check(done['status']=='completed' and set(done['completed_fits'])==FITS and len(done['completed_fits'])==9,'Nine completed fits')
    names={'started.json','plan.json','references.npz'}|{f'orders-{s}.npy' for s in SEEDS}|{f'fits/{fit}/{name}' for fit in FITS for name in ('completed.json','updates.jsonl','weights.pt','dev-predictions.npz')}
    manifest(RUN,done['files'],names)
    for name,pin in done['source_sha256'].items(): bind(ROOT/name,pin)
    prepared=read(PREP/'completed.json'); manifest(PREP,prepared['files'],{'started.json','catalog.json','rows.jsonl','summary.json'})
    return read(DIAG/'summary.json'),original,receipt


def classify(candidate_id,value,boolean):
    if candidate_id=='reserved:NOT_MENTIONED': return 'none'
    if candidate_id=='reserved:DONTCARE': return 'dontcare'
    return value.strip().casefold() if boolean else 'other'


def load_metadata():
    catalog=read(PREP/'catalog.json')['queries']
    with (PREP/'rows.jsonl').open() as f: ledger=[decode(line) for line in f]
    rows={'train':[],'dev':[]}; classes={'train':[],'dev':[]}
    for index,r in enumerate(ledger):
        check(type(r['row_index']) is int and r['row_index']==index,'Complete row identity')
        if r['admission']!='admitted': continue
        q=catalog[r['query_index']]; ids=q['candidate_ids']; target=r['current_label_index']; previous=r['previous_current_index']; old=ledger[r['previous_row_index']]
        check(all(q[k]==r[k] for k in ('split','query_id','service','slot')),'Schema row join')
        check(type(target) is int and type(previous) is int and 0<=target<len(ids) and 0<=previous<len(ids) and len(ids)==r['candidate_count'],'Canonical support')
        check(ids[target]==r['current_candidate_id'] and ids[previous]==r['previous_candidate_id']==old['current_candidate_id'],'Canonical target/prior IDs')
        check(old['time']==r['time']-1 and all(old[k]==r[k] for k in ('split','dialogue_id','service','slot')),'Adjacent predecessor')
        kinds=[classify(cid,value,q['boolean_slot']) for cid,value in zip(ids,q['candidate_values'],strict=True)]
        check(all(k in VALUES for k in kinds) and kinds[target]==r['current_value_group'],'Canonical value classes')
        rows[r['split']].append(r); classes[r['split']].append(kinds)
    check(len(rows['train'])==42810 and len(rows['dev'])==51070,'Previously audited admitted membership')
    return rows,classes


def denom(rows,indices):
    return {'rows':len(indices),'dialogues':len({rows[i]['dialogue_id'] for i in indices}),'distinct_schema_queries':len({rows[i]['query_id'] for i in indices}),'dialogue_query_streams':len({(rows[i]['dialogue_id'],rows[i]['query_id']) for i in indices})}


def grouped(rows):
    groups={}
    for panel in ('seen','unseen'):
        ix=[i for i,r in enumerate(rows) if r['unseen']==(panel=='unseen')]
        groups[panel+'/all']=ix
        for changed,name in [(True,'changed'),(False,'retained')]: groups[panel+'/'+name]=[i for i in ix if (rows[i]['derived_bin'] in BINS[2:])==changed]
        for t in BINS: groups[panel+'/transition/'+t]=[i for i in ix if rows[i]['derived_bin']==t]
        for v in VALUES:
            vi=[i for i in ix if rows[i]['current_value_group']==v]
            groups[panel+'/value/'+v]=vi
            for changed,name in [(True,'changed'),(False,'retained')]: groups[panel+'/value/'+v+'/'+name]=[i for i in vi if (rows[i]['derived_bin'] in BINS[2:])==changed]
            for t in BINS: groups[panel+'/transition_value/'+t+'/'+v]=[i for i in vi if rows[i]['derived_bin']==t]
    check(len(groups)==96,'Fixed exhaustive 96 groups')
    return groups


def training(rows,classes):
    check(all(r['split']=='train' and r['admission']=='admitted' for r in rows),'Training-only support')
    result={'rows':len(rows),'transition_value':{},'previous_to_current_value':{}}
    for t in BINS:
        for v in VALUES: result['transition_value'][t+'/'+v]=denom(rows,[i for i,r in enumerate(rows) if r['derived_bin']==t and r['current_value_group']==v])
    for p in VALUES:
        for c in VALUES: result['previous_to_current_value'][p+'/'+c]=denom(rows,[i for i,r in enumerate(rows) if classes[i][r['previous_current_index']]==p and r['current_value_group']==c])
    check(sum(v['rows'] for v in result['transition_value'].values())==len(rows)==sum(v['rows'] for v in result['previous_to_current_value'].values()),'Exhaustive train support')
    return result


def mean(values,ix): return math.fsum(float(values[i]) for i in ix)/len(ix) if ix else None


def fit(rows,classes,name,groups):
    path=RUN/'fits'/name/'dev-predictions.npz'
    with np.load(path,allow_pickle=False) as z:
        check(set(z.files)=={'row_indices','log_probs'},'Prediction fields')
        log=z['log_probs']; row_ids=z['row_indices']
    check(log.dtype==np.float32 and log.shape==(len(rows),12) and row_ids.dtype==np.int64 and np.array_equal(row_ids,[r['row_index'] for r in rows]),'Full prediction alignment')
    valid=np.arange(12)[None,:]<np.asarray([r['candidate_count'] for r in rows])[:,None]
    check(np.isfinite(log[valid]).all() and (log[valid]<=0).all() and np.isneginf(log[~valid]).all(),'Supported finite log probabilities')
    logs=log.astype(np.float64); probs=np.exp(logs)
    check((np.abs(probs.sum(1)-1)<=2e-6).all(),'Unmodified float64 probability mass')
    target=np.asarray([r['current_label_index'] for r in rows]); prior=np.asarray([r['previous_current_index'] for r in rows]); choice=logs.argmax(1)
    changed=target!=prior; correct=choice==target; prior_error=changed & (choice==prior); other_wrong=(choice!=target)&((~changed)|(choice!=prior))
    check(np.all(correct.astype(int)+prior_error.astype(int)+other_wrong.astype(int)==1),'Disjoint row outcomes')
    index=np.arange(len(rows)); q={'target_probability':probs[index,target],'prior_probability':probs[index,prior],'target_nll':-logs[index,target]}
    q['strongest_other_probability']=np.asarray([max(probs[i,j] for j in range(r['candidate_count']) if j!=target[i] and j!=prior[i]) for i,r in enumerate(rows)])
    margin=logs[index,target]-logs[index,prior]
    choices=[classes[i][int(choice[i])] for i in range(len(rows))]
    ties=[sum(logs[i,j]==logs[i,choice[i]] for j in range(r['candidate_count']))>1 for i,r in enumerate(rows)]
    cells={}
    for group,ix in groups.items():
        ci=[i for i in ix if changed[i]]
        cell=denom(rows,ix)
        cell.update(changed_rows=len(ci),retained_rows=len(ix)-len(ci),prediction_events=len(ix),correct_count=sum(int(correct[i]) for i in ix),prior_error_count=sum(int(prior_error[i]) for i in ix),other_wrong_count=sum(int(other_wrong[i]) for i in ix),top1_tied_count=sum(int(ties[i]) for i in ix))
        cell.update({'mean_'+k:mean(v,ix) for k,v in q.items()})
        cell['changed_mean_target_minus_prior_log_probability']=mean(margin,ci)
        wrong=Counter(choices[i] for i in ix if not correct[i]); cell['wrong_choice_classes']={v:wrong[v] for v in VALUES}
        check(cell['correct_count']+cell['prior_error_count']+cell['other_wrong_count']==len(ix),'Group outcome reconciliation')
        check(sum(wrong.values())==len(ix)-cell['correct_count'],'Wrong-choice class reconciliation')
        cells[group]=cell
    return cells,q['target_nll']


def decomposition(rows,nll,original):
    primary=[i for i,r in enumerate(rows) if r['unseen'] and r['derived_bin'] in BINS[2:]]; n=len(primary)
    check(n==original['primary']['rows']==1704,'Primary common denominator')
    result={'primary_rows':n,'groups':{}}
    for v in VALUES:
        ix=[i for i in primary if rows[i]['current_value_group']==v]; g=denom(rows,ix); g['seeds']=[]
        for seed in SEEDS:
            s=nll[f'slot-{seed}']; c=nll[f'candidate-{seed}']; differences=[float(c[i])-float(s[i]) for i in ix]
            g['seeds'].append({'seed':seed,'slot_global_denominator_nll_contribution':math.fsum(float(s[i]) for i in ix)/n,'candidate_global_denominator_nll_contribution':math.fsum(float(c[i]) for i in ix)/n,'candidate_minus_slot_global_denominator_contribution':math.fsum(differences)/n,'conditional_mean_difference':math.fsum(differences)/len(ix) if ix else None})
        g['mean_candidate_minus_slot_global_denominator_contribution']=math.fsum(s['candidate_minus_slot_global_denominator_contribution'] for s in g['seeds'])/3
        result['groups'][v]=g
    reconstructed=[]
    for j,seed in enumerate(SEEDS):
        value=math.fsum(g['seeds'][j]['candidate_minus_slot_global_denominator_contribution'] for g in result['groups'].values())
        expected=original['primary']['paired'][j]
        check(seed==expected['seed'] and abs(value-expected['candidate_minus_slot_nll'])<1e-12,'Published paired decomposition')
        reconstructed.append(value)
    check(abs(math.fsum(reconstructed)/3-original['primary']['mean_candidate_minus_slot_nll'])<1e-12,'Published pooled decomposition')
    result['reconstructed_paired_differences']=reconstructed
    return result


def write(name,value):
    with (OUT/name).open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def main():
    check({p.name for p in OUT.iterdir()}=={'audit.py'},'Exclusive unexecuted audit directory required')
    started=time.monotonic(); source=sha(__file__)
    try:
        published,original,diagnostic_receipt=authenticate(); rows,classes=load_metadata(); groups=grouped(rows['dev'])
        result={'training_support':training(rows['train'],classes['train']),'fits':{},'groups':{name:{**denom(rows['dev'],ix),'prediction_events_across_nine_fits':9*len(ix)} for name,ix in groups.items()}}
        nll={}
        check(set(published['fits'])==FITS,'Nine diagnostic fits')
        for name in sorted(FITS): result['fits'][name],nll[name]=fit(rows['dev'],classes['dev'],name,groups)
        result['primary_decomposition']=decomposition(rows['dev'],nll,original)
        for key in result: compare(published[key],result[key],key)
        check(published['original_continuation_allowed'] is False and published['new_primary_or_continuation_rule'] is False,'No changed study decision')
        for name,item in INPUTS.items():
            p=ROOT/name; check(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'End input stability '+name)
        result.update(status='completed',all_diagnostic_fields_recomputed_match=True,fit_count=9,groups_per_fit=96,verified_fit_group_cells=864,original_continuation_allowed=False,diagnostic_summary_sha256=SUMMARY_SHA,original_run_completed_sha256=RUN_SHA,original_analysis_sha256=ORIGINAL_SHA,recorded_diagnostic_wall_seconds=diagnostic_receipt['wall_seconds'],scope_limits=['No diagnostic, reporter, producer, model or encoder modules imported. No inference, optimizer, RNG draws or probability repair.','All integer identities/partitions exact. Float64 probability/log arithmetic independently accumulated with math.fsum, compared with 1e-12 absolute/relative arithmetic tolerance.','Canonical prepared metadata and full training execution validity inherit the previously authenticated independent audits; all current run payload hashes, metadata identities and diagnostic source Git identities were rechecked.','All rows and all nine fits retained; per-fit events and nine-fit events are not independent data points. Raw predictions and label rows are not published.','Actual clocks, model-call counters and remote publication chronology remain authenticated receipt witnesses, not independently observed.','Posthoc descriptive decomposition cannot establish causality, missing information, rollout performance or a new successful continuation decision.'])
        write('summary.json',result)
        write('receipt.json',{'status':'completed','source_sha256':source,'source_file':'audit.py','files':{'audit.py':{'sha256':source,'bytes':Path(__file__).stat().st_size},'summary.json':{'sha256':sha(OUT/'summary.json'),'bytes':(OUT/'summary.json').stat().st_size}},'authenticated_inputs':INPUTS,'diagnostic_summary_sha256':SUMMARY_SHA,'audit_wall_seconds':time.monotonic()-started,'new_model_calls':0,'new_training_calls':0,'new_rng_calls':0,'original_continuation_allowed':False})
        print(json.dumps({'status':'completed','source_sha256':source,'summary_sha256':sha(OUT/'summary.json'),'receipt_sha256':sha(OUT/'receipt.json'),'fit_group_cells':864}))
    except BaseException as e:
        try: write('failed.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'source_sha256':source,'diagnostic_summary_sha256':SUMMARY_SHA,'wall_seconds':time.monotonic()-started})
        except BaseException as secondary:
            if callable(getattr(e,'add_note',None)): e.add_note('Failure preservation error: '+repr(secondary))
        raise

if __name__=='__main__': main()
