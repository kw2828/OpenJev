"""Independent saved-output audit. Never import experiment/report/model modules.

Invoke only after all nine fits and the frozen reporter have completed. If the
producer failed, use --failure-only, which never opens prediction arrays.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
RUN=ROOT/'runs/dialogue-conditional-v1/training-01'
REPORT=ROOT/'output/dialogue-conditional-v1/analysis-01'
FREEZE=ROOT/'output/dialogue-conditional-v1/training-protocol-01'
PREP=ROOT/'runs/dialogue-conditional-v1/preparation-01'
PLAN_SHA='0bc43fadc5cd4bdb1055810c0a415dae9bc6930ae45129b2bee6774b82400835'
COMMIT='55526c5ace9470b50ed1693c989359a84f5989a3'
PREP_SHA='960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3'
PREP_AUDIT=ROOT/'output/dialogue-conditional-v1/preparation-audit-01'
PREP_AUDIT_RECEIPT='e0978860a1eed2527ab53d44f235cb69ba28858f38c5fc7a6143fbc6eb89b17b'
PREP_AUDIT_SUMMARY='94dba92c19bbc424b316cbc37dd8d2bb1454c171b478e29d967a87207c4600b8'
SEEDS=(5301,5302,5303); MODES=('mean','slot','candidate')
ORDER=[f'{m}-{s}' for s,ms in zip(SEEDS,(MODES,('slot','candidate','mean'),('candidate','mean','slot')),strict=True) for m in ms]
BINS=('unmentioned_retention','assigned_retention','first_assignment','revision','clear')
VALUES=('none','true','false','dontcare','other')
SOURCES={'research/dialogue-conditional-observation-design.md','research/dialogue-conditional-preparation-protocol.md','scripts/prepare_dialogue_conditional.py','scripts/prepare_dialogue_tokens.py','scripts/study_dialogue_copy.py','scripts/study_dialogue_memory.py','scripts/study_dialogue_tokens.py','src/openjev/research/dialogue_copy_features.py','src/openjev/research/dialogue_state_data.py','tests/test_prepare_dialogue_conditional.py','src/openjev/research/dialogue_conditional_observation.py','tests/test_dialogue_conditional_observation.py','src/openjev/research/dialogue_copy_memory.py','scripts/study_dialogue_conditional.py','tests/test_study_dialogue_conditional.py','scripts/report_dialogue_conditional.py','tests/test_report_dialogue_conditional.py','research/dialogue-conditional-training-protocol.md'}
CONFIG={'methods':list(MODES),'seeds':list(SEEDS),'epochs':20,'batch_size':256,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,'input_dim':384,'projection_dim':64,'hidden_dim':64,'threads':4,'interop_threads':1,'dtype':'float32','deterministic':True}
LIMITS={'wall_seconds':3600.,'rss_bytes':6*1024**3,'output_bytes':512*1024**2}
LEXICAL=['user_match','system_match','unique_longest_user','unique_longest_system','literal_current','literal_previous','is_none','is_dontcare','affirmative_cue_for_true','negative_cue_for_false']
INPUTS={}


def need(ok,label):
    if not ok: raise ValueError(label)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def decode(text):
    def pairs(xs):
        d={}
        for k,v in xs:
            need(k not in d,'Repeated JSON field'); d[k]=v
        return d
    def invalid(x): raise ValueError('Invalid JSON scalar '+x)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def read(p): return decode(Path(p).read_text())


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def equal(x,y,label): need(canonical(x)==canonical(y),label)


def bind(p,pin,size=None):
    p=Path(p).resolve(); need(p.is_relative_to(ROOT) and p.is_file(),'Input boundary')
    name=p.relative_to(ROOT).as_posix()
    if name not in INPUTS: INPUTS[name]={'sha256':sha(p),'bytes':p.stat().st_size}
    got=INPUTS[name]
    need(got['sha256']==pin and (size is None or got['bytes']==size),'Input digest/bytes: '+name)
    return got


def child(base,name):
    p=Path(name); need(not p.is_absolute() and '..' not in p.parts,'Unsafe member')
    out=base/p; need(not out.is_symlink() and out.resolve().is_relative_to(base.resolve()),'Escaping member')
    return out


def manifest(folder,mapping,expected):
    need(set(mapping)==set(expected),'Manifest members')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    need(actual==set(expected)|{'completed.json'},'Exact tree membership')
    for name,item in mapping.items():
        need(set(item)=={'sha256','bytes'} and type(item['bytes']) is int,'Payload record')
        bind(child(folder,name),item['sha256'],item['bytes'])


def read_npz(path,names):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(names),'Array payload fields')
        return {k:z[k] for k in names}


def finite(v): return type(v) in (int,float) and math.isfinite(v)


def freeze_identity():
    bind(FREEZE/'plan.json',PLAN_SHA)
    need(hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+(FREEZE/'plan.json').relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()==PLAN_SHA,'Published plan bytes')
    plan=read(FREEZE/'plan.json'); f=read(FREEZE/'completed.json')
    need(f['status']=='completed' and f['phase']=='freeze' and f['plan_sha256']==PLAN_SHA,'Freeze status')
    equal(plan['config'],CONFIG,'Fixed recipe'); equal(plan['limits'],LIMITS,'Fixed caps')
    need(plan['version']=='dialogue-conditional-training-v1' and plan['expected_fits']==ORDER and set(plan['source_sha256'])==SOURCES,'Frozen study identity')
    names={'started.json','plan.json','prepared-completed.json'}|{f'orders-{s}.npy' for s in SEEDS}|{'sources/'+n for n in SOURCES}
    manifest(FREEZE,f['files'],names)
    for name,dig in plan['source_sha256'].items():
        bind(ROOT/name,dig); bind(FREEZE/'sources'/name,dig)
        need(hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)).hexdigest()==dig,'Published source '+name)
    need(subprocess.run(['git','merge-base','--is-ancestor',COMMIT,'HEAD'],cwd=ROOT,capture_output=True).returncode==0,'Local Git ancestry')
    need(plan['prepared_completed_sha256']==PREP_SHA and Path(plan['prepared_path']).resolve()==PREP,'Prepared fixed identity')
    bind(PREP/'completed.json',PREP_SHA); bind(FREEZE/'prepared-completed.json',PREP_SHA)
    bind(PREP_AUDIT/'receipt.json',PREP_AUDIT_RECEIPT); bind(PREP_AUDIT/'summary.json',PREP_AUDIT_SUMMARY)
    a=read(PREP_AUDIT/'summary.json'); ar=read(PREP_AUDIT/'receipt.json')
    need(a['status']=='completed' and a['all_saved_rows_and_aggregates_match'] is True and a['preparation_completed_sha256']==PREP_SHA and ar['all_metadata_checks_passed'] is True,'Inherited independent metadata audit')
    need(ar['files']['summary.json']['sha256']==PREP_AUDIT_SUMMARY,'Inherited audit summary identity')
    prepared=read(PREP/'completed.json')
    manifest(PREP,prepared['files'],{'started.json','catalog.json','rows.jsonl','summary.json'})
    equal(plan['prepared_files'],prepared['files'],'Prepared file closure')
    # Float payloads below are hashed opaquely. Only lexical literal_current is numerically read later.
    for name,item in prepared['authenticated_inputs'].items(): bind(name,item['sha256'],item['bytes'])
    return plan,prepared


def metadata(plan):
    with (PREP/'rows.jsonl').open() as f: all_rows=[decode(x) for x in f]
    catalog=read(PREP/'catalog.json')['queries']; rows={'train':[],'dev':[]}
    for i,r in enumerate(all_rows):
        need(type(r['row_index']) is int and r['row_index']==i,'Prepared original row order')
        q=catalog[r['query_index']]; ids=q['candidate_ids']; y=r['current_label_index']
        equal([r[k] for k in ('split','query_id','service','slot')],[q[k] for k in ('split','query_id','service','slot')],'Catalog join')
        need(type(y) is int and 0<=y<len(ids) and ids[y]==r['current_candidate_id'] and len(ids)==r['candidate_count'],'Current canonical ID')
        category='none' if ids[y]=='reserved:NOT_MENTIONED' else 'dontcare' if ids[y]=='reserved:DONTCARE' else q['candidate_values'][y].strip().casefold() if q['boolean_slot'] else 'other'
        need(category==r['current_value_group'] and r['boolean_slot']==q['boolean_slot'],'Value category')
        if r['admission']!='admitted': continue
        p=r['previous_current_index']; old=all_rows[r['previous_row_index']]
        need(type(p) is int and 0<=p<len(ids) and ids[p]==r['previous_candidate_id']==old['current_candidate_id'],'Prior canonical remap')
        need(old['time']==r['time']-1 and all(old[k]==r[k] for k in ('split','dialogue_id','service','slot')),'Adjacent prior')
        if ids[p]==ids[y]: b='unmentioned_retention' if ids[y]=='reserved:NOT_MENTIONED' else 'assigned_retention'
        elif ids[p]=='reserved:NOT_MENTIONED': b='first_assignment'
        elif ids[y]=='reserved:NOT_MENTIONED': b='clear'
        else: b='revision'
        need(b==r['derived_bin'] and type(r['unseen']) is bool,'Transition/panel identity')
        rows[r['split']].append(r)
    need(len(rows['train'])==plan['admitted_train_rows'] and len(rows['dev'])==plan['admitted_dev_rows'],'Complete admitted cohort')
    return rows


def geometry(batch,mode):
    b=len(batch); c=max(r['candidate_count'] for r in batch)
    lens=[r['cache']['token_stop']-r['cache']['token_start'] for r in batch]
    l=0 if mode=='mean' else max(lens)
    shapes={'observation':[b,384] if mode=='mean' else [b,l,384],'query':[b,384],'candidates':[b,c,384],'candidate_mask':[b,c],'lexical':[b,c,10],'previous_onehot':[b,c]}
    if mode!='mean': shapes.update(token_mask=[b,l],token_prior=[b,l])
    floats=sum(math.prod(v) for k,v in shapes.items() if k not in ('candidate_mask','token_mask'))
    work={'rows':b,'supported_candidate_positions':sum(r['candidate_count'] for r in batch),'padded_candidate_positions':b*c,'supported_token_positions':0 if mode=='mean' else sum(lens),'padded_token_positions':b*l,'attention_score_positions':b*c*l,'token_key_positions':b*l,'attention_query_positions':0 if mode=='mean' else b*c,'scorer_positions':b*c,'float_input_scalars':floats,'float_input_bytes':4*floats,'boolean_input_bytes':b*c+b*l,'max_token_length':l,'max_candidate_count':c}
    return work,shapes


def norm_check(n,batches,rows,work):
    equal([n[k] for k in ('batches','rows','supported_candidates','masked_candidates')],[batches,rows,work['supported_candidate_positions'],work['padded_candidate_positions']-work['supported_candidate_positions']],'Normalization coverage')
    need(finite(n['max_abs_mass_error']) and 0<=n['max_abs_mass_error']<=2e-6 and finite(n['min_supported_log_prob']) and finite(n['max_supported_log_prob']) and n['min_supported_log_prob']<=n['max_supported_log_prob']<=0,'Recorded normalization extrema')


def verify_training(done,plan,rows):
    n,d=len(rows['train']),len(rows['dev']); u=20*((n+255)//256); ev=(d+255)//256
    need(plan['updates_per_fit']==u and plan['evaluation_batches_per_fit']==ev,'Full fit budgets')
    counts=Counter(0 if r['derived_bin']==BINS[0] else 1 if r['derived_bin']==BINS[1] else 2 for r in rows['train'])
    equal(plan['objective'],{'counts':[counts[i] for i in range(3)],'weights':[n/(3*counts[i]) for i in range(3)]},'Admitted training-only weights')
    per={'forward_attempted':u+ev,'forward_returned':u+ev,'backward_attempted':u,'backward_returned':u,'optimizer_attempted':u,'optimizer_returned':u,'training_rows':20*n,'evaluation_rows':d}
    equal(done['progress']['totals'],{k:9*v for k,v in per.items()},'Complete attempted/returned totals')
    need(done['progress']['completed_fits']==ORDER and done['progress']['active_fit'] is None,'Terminal completed coverage')
    recs={}; phase_totals=Counter(); fit_wall=0.
    for seed in SEEDS:
        spec=plan['orders'][str(seed)]; need(spec['file']==f'orders-{seed}.npy','Orders filename')
        bind(RUN/spec['file'],spec['sha256']); bind(FREEZE/spec['file'],spec['sha256'])
        orders=np.load(RUN/spec['file'],allow_pickle=False)
        need(orders.dtype==np.int64 and orders.shape==(20,n),'Orders geometry')
        need(all(np.array_equal(np.sort(x),np.arange(n,dtype=np.int64)) for x in orders),'Every epoch is complete permutation')
        for mode in MODES:
            name=f'{mode}-{seed}'; folder=RUN/'fits'/name; r=read(folder/'completed.json')
            need(r['status']=='completed' and r['method']==mode and r['seed']==seed and r['epochs']==20 and r['plan_sha256']==PLAN_SHA and r['orders_sha256']==spec['sha256'],'Fit identity')
            equal(r['counts'],per,'Fit operation coverage'); manifest(folder,r['files'],{'updates.jsonl','weights.pt','dev-predictions.npz'})
            cfg=r['configuration']; expected={'class':'DialogueConditionalObservation','version':'dialogue-conditional-observation-v1','mode':mode,'input_dim':384,'projection_dim':64,'hidden_dim':64,'feature_dim':396,'lexical_fields':LEXICAL,'parameters':99393 if mode=='mean' else 173121,'shared_scorer_parameters':99393,'attention_parameters':0 if mode=='mean' else 73728,'zero_input_parameters':64,'softmax_shift_parameters':1,'attention_width':None if mode=='mean' else 64,'schema_pair':None if mode=='mean' else '[query;query]' if mode=='slot' else '[query;candidate]'}
            equal({k:cfg[k] for k in expected},expected,'Model configuration witness')
            for key in ('initial_common_sha256','initial_attention_sha256'):
                value=r[key]
                if key=='initial_attention_sha256' and mode=='mean': need(value is None,'No mean attention')
                else: need(type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value),'Initializer digest')
            need(0<r['evaluation']['wall_seconds']<=r['wall_seconds']<=done['wall_seconds'] and 0<r['process_lifetime_peak_rss_bytes']<=done['process_lifetime_peak_rss_bytes'],'Fit cost witness')
            work_total=Counter(); norms=Counter(); mins=[]; maxs=[]; mass=[]; local_phases=Counter(); k=0; max_bytes=0
            with (folder/'updates.jsonl').open() as f:
                for epoch,order in enumerate(orders):
                    for start in range(0,n,256):
                        line=f.readline(); need(bool(line),'Missing update')
                        event=decode(line); batch=[rows['train'][int(i)] for i in order[start:start+256]]; k+=1
                        equal([event['epoch'],event['start'],event['update'],event['row_indices']],[epoch,start,k,[x['row_index'] for x in batch]],'Paired update membership/order')
                        w,shapes=geometry(batch,mode); equal(event['work'],w,'Independent update work'); equal(event['actor_shapes'],shapes,'Independent actor geometry')
                        need(finite(event['weighted_loss']) and event['weighted_loss']>=0,'Finite weighted loss witness')
                        need(set(event['phase_seconds'])=={'assembly','forward_validation_loss','backward_clip','optimizer'} and all(finite(v) and v>=0 for v in event['phase_seconds'].values()),'Phase timing fields')
                        local_phases.update(event['phase_seconds']); work_total.update({a:b for a,b in w.items() if not a.startswith('max_')}); max_bytes=max(max_bytes,w['float_input_bytes'])
                        z=event['normalization']; norm_check(z,1,len(batch),w)
                        norms.update({a:z[a] for a in ('batches','rows','supported_candidates','masked_candidates')}); mins.append(z['min_supported_log_prob']); maxs.append(z['max_supported_log_prob']); mass.append(z['max_abs_mass_error'])
                need(f.read()=='','Extra optimizer update')
            norms.update(max_abs_mass_error=max(mass),min_supported_log_prob=min(mins),max_supported_log_prob=max(maxs))
            equal(dict(norms),r['training_normalization'],'Training invariant reconciliation'); equal(dict(work_total),r['training_work'],'Training work totals')
            equal(plan['work_schedules'][name]['training'],{'batches':u,'totals':dict(work_total),'maximum_batch_float_input_bytes':max_bytes},'Frozen full training schedule')
            ew=Counter(); emax=0
            for start in range(0,d,256):
                w,_=geometry(rows['dev'][start:start+256],mode); ew.update({a:b for a,b in w.items() if not a.startswith('max_')}); emax=max(emax,w['float_input_bytes'])
            equal(dict(ew),r['evaluation']['work'],'Independent evaluation work'); need(r['evaluation']['rows']==d,'Evaluation coverage')
            equal(plan['work_schedules'][name]['evaluation'],{'batches':ev,'totals':dict(ew),'maximum_batch_float_input_bytes':emax},'Frozen full evaluation schedule')
            norm_check(r['evaluation']['normalization'],ev,d,ew)
            need(sum(local_phases.values())<=r['wall_seconds'],'Disjoint update phases fit inside fit wall')
            phase_totals.update(local_phases); fit_wall+=r['wall_seconds']; recs[name]=r
        need(len({recs[f'{m}-{seed}']['initial_common_sha256'] for m in MODES})==1 and recs[f'slot-{seed}']['initial_attention_sha256']==recs[f'candidate-{seed}']['initial_attention_sha256'],'Paired initializer witnesses')
    need(fit_wall<=done['wall_seconds'],'Fit wall sum inside whole wall')
    equal(done['fits'],[{'method':recs[f]['method'],'seed':recs[f]['seed'],'counts':per} for f in ORDER],'Root fit metadata')
    return recs,{'updates_per_fit':u,'total_optimizer_updates':9*u,'evaluation_batches_per_fit':ev,'sum_fit_wall_seconds':fit_wall,'sum_update_phase_seconds':dict(phase_totals)}


def memberships(rows):
    out={}
    for panel in ('seen','unseen'):
        eligible=[i for i,r in enumerate(rows) if r['unseen']==(panel=='unseen')]
        definitions=['all','changed','retained']+['transition/'+b for b in BINS]
        definitions += [name for v in VALUES for name in ('value/'+v,'value/'+v+'/changed','value/'+v+'/retained')]
        for name in definitions:
            selected=[]
            for i in eligible:
                r=rows[i]; changed=r['derived_bin'] in BINS[2:]; pieces=name.split('/')
                yes=name=='all' or name=='changed' and changed or name=='retained' and not changed
                if pieces[0]=='transition': yes=r['derived_bin']==pieces[1]
                elif pieces[0]=='value': yes=r['current_value_group']==pieces[1] and (len(pieces)==2 or changed==(pieces[2]=='changed'))
                if yes: selected.append(i)
            out[panel+'/'+name]=selected
    return out


def cells(rows,indexes,values):
    result={}
    for name,ix in indexes.items():
        ds={rows[i]['dialogue_id'] for i in ix}
        cell={'rows':len(ix),'dialogues':len(ds),'distinct_schema_queries':len({rows[i]['query_id'] for i in ix}),'dialogue_query_streams':len({(rows[i]['dialogue_id'],rows[i]['query_id']) for i in ix})}
        for metric,v in values.items():
            per=defaultdict(list)
            for i in ix: per[rows[i]['dialogue_id']].append(float(v[i]))
            cell[metric]=math.fsum(float(v[i]) for i in ix)/len(ix) if ix else None
            cell['equal_dialogue_'+metric]=math.fsum(math.fsum(x)/len(x) for x in per.values())/len(per) if per else None
        result[name]=cell
    return result


def probabilities(rows,z):
    ids=np.asarray([r['row_index'] for r in rows],np.int64); lp=z['log_probs']
    need(z['row_indices'].dtype==np.int64 and np.array_equal(z['row_indices'],ids),'Prediction row identity')
    need(lp.dtype==np.float32 and lp.shape==(len(rows),12),'Prediction matrix')
    support=np.arange(12)[None,:]<np.asarray([r['candidate_count'] for r in rows])[:,None]
    need(np.isfinite(lp[support]).all() and (lp[support]<=0).all() and np.isneginf(lp[~support]).all(),'Saved log support')
    p=np.exp(lp.astype(np.float64)); error=np.abs(np.sum(p,axis=1)-1)
    need((error<=2e-6).all(),'Independent float64 mass reconstruction')
    y=np.asarray([r['current_label_index'] for r in rows]); correct=(np.argmax(lp,axis=1)==y).astype(np.float64)
    nll=-lp[np.arange(len(rows)),y].astype(np.float64)
    residual=p.copy(); residual[np.arange(len(rows)),y]-=1
    return {'accuracy':correct,'nll':nll,'brier':np.einsum('ij,ij->i',residual,residual)},float(error.max()),float(lp[support].min()),float(lp[support].max())


def compare(actual,expected,path='root'):
    if type(expected) is dict:
        need(type(actual) is dict and set(actual)==set(expected),'Metric field membership '+path)
        for k in expected: compare(actual[k],expected[k],path+'/'+str(k))
    elif type(expected) is list:
        need(type(actual) is list and len(actual)==len(expected),'Metric list '+path)
        for i,(a,b) in enumerate(zip(actual,expected,strict=True)): compare(a,b,path+'/'+str(i))
    elif type(expected) is float:
        need(type(actual) in (float,int) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),'Metric arithmetic '+path)
    else: need(type(actual) is type(expected) and actual==expected,'Metric identity '+path)


def analyze(rows,recs,plan,prepared):
    indexes=memberships(rows); fitted={}
    for seed in SEEDS:
        for mode in MODES:
            name=f'{mode}-{seed}'; z=read_npz(RUN/'fits'/name/'dev-predictions.npz',('row_indices','log_probs'))
            values,err,lo,hi=probabilities(rows,z)
            need(lo==recs[name]['evaluation']['normalization']['min_supported_log_prob'] and hi==recs[name]['evaluation']['normalization']['max_supported_log_prob'],'Saved evaluation log extrema')
            fitted[name]={'rows':len(rows),'maximum_raw_probability_mass_error':err,'cells':cells(rows,indexes,values)}
    refs=read_npz(RUN/'references.npz',('row_indices','previous_indices','literal_indices'))
    need(refs['row_indices'].dtype==np.int64 and np.array_equal(refs['row_indices'],[r['row_index'] for r in rows]),'Reference row alignment')
    lexical_path=Path(plan['feature_headers']['lexical']['path']).resolve(); pin=prepared['authenticated_inputs'][str(lexical_path)]
    bind(lexical_path,pin['sha256'],pin['bytes'])
    a=np.load(lexical_path,mmap_mode='r',allow_pickle=False)
    need(a.dtype==np.float32 and not a.flags.writeable and list(a.shape)==plan['feature_headers']['lexical']['shape'],'Literal-only read-only array')
    prev=[]; literal=[]
    for r in rows:
        start=r['cache']['lexical_start']; c=r['candidate_count']; v=a[start:start+10*c].reshape(c,10)[:,4]
        need(np.isin(v,[0.,1.]).all() and (v==1).sum()==1,'Canonical literal one-hot')
        literal.append(int(np.flatnonzero(v==1)[0])); prev.append(r['previous_current_index'])
    reference={}; target=np.asarray([r['current_label_index'] for r in rows])
    for name,key,choices in [('previous_gold_carry','previous_indices',prev),('literal_carry','literal_indices',literal)]:
        need(refs[key].dtype==np.int64 and refs[key].shape==target.shape and np.array_equal(refs[key],choices),'Independent reference reconstruction')
        reference[name]={'cells':cells(rows,indexes,{'accuracy':(np.asarray(choices)==target).astype(np.float64)}),'scope':'Deterministic reference accuracy only; no finite probability loss assigned'}
    support=fitted['slot-5301']['cells']['unseen/changed']; paired=[]
    for seed in SEEDS:
        slot=fitted[f'slot-{seed}']['cells']['unseen/changed']['nll']; candidate=fitted[f'candidate-{seed}']['cells']['unseen/changed']['nll']
        delta=candidate-slot if support['rows'] else None
        paired.append({'seed':seed,'slot_nll':slot,'candidate_nll':candidate,'candidate_minus_slot_nll':delta,'relative_nll_change':delta/slot if support['rows'] and slot>0 else None})
    delta=math.fsum(p['candidate_minus_slot_nll'] for p in paired)/3 if support['rows'] else None
    base=math.fsum(p['slot_nll'] for p in paired)/3 if support['rows'] else None
    passed=bool(support['rows'] and delta<0 and all(p['candidate_minus_slot_nll']<0 for p in paired))
    primary={'panel':'unseen/changed','weighting':'row mean within fit, paired difference then three-seed mean',**{k:support[k] for k in ('rows','dialogues','distinct_schema_queries','dialogue_query_streams')},'paired':paired,'mean_candidate_minus_slot_nll':delta,'relative_mean_nll_change':delta/base if support['rows'] and base>0 else None,'all_three_seed_differences_negative':bool(support['rows'] and all(p['candidate_minus_slot_nll']<0 for p in paired)),'primary_nll_rule_passed':passed}
    return fitted,reference,primary


def authenticate_done(pin,plan):
    bind(RUN/'completed.json',pin); d=read(RUN/'completed.json')
    need(d['status']=='completed' and d['phase']=='train' and d['version']=='dialogue-conditional-training-v1' and d['no_retry'] is True and d['completed_fits']==ORDER and d['expected_fits']==ORDER,'All nine fits completed')
    need(d['plan_sha256']==PLAN_SHA and d['prepared_completed_sha256']==PREP_SHA and d['quality_metrics_computed'] is False and d['encoder_calls']==0 and d['test_contents_accessed'] is False,'Completed scope')
    equal(d['runtime'],plan['runtime'],'Runtime witness'); equal(d['source_sha256'],plan['source_sha256'],'Execution sources')
    names={'started.json','plan.json','references.npz'}|{f'orders-{s}.npy' for s in SEEDS}|{f'fits/{name}/{file}' for name in ORDER for file in ('completed.json','weights.pt','updates.jsonl','dev-predictions.npz')}
    manifest(RUN,d['files'],names); bind(RUN/'plan.json',PLAN_SHA)
    request=read(RUN/'started.json')['request']; need(request['plan_sha256']==PLAN_SHA and Path(request['plan']).resolve()==FREEZE/'plan.json','Execution request')
    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    need(0<d['wall_seconds']<=3600 and 0<d['process_lifetime_peak_rss_bytes']<=6*1024**3 and size<=512*1024**2,'Whole run cost caps')
    return d,size


def inherited_costs(prepared):
    result={}
    for name,relative in {'pooled':'runs/sgd-state-v1/features-02','lexical':'runs/dialogue-copy-v1/lexical-01','tokens':'runs/dialogue-token-v1/features-01'}.items():
        folder=ROOT/relative; ident=prepared['authenticated_inputs'][str(folder/'completed.json')]
        bind(folder/'completed.json',ident['sha256'],ident['bytes']); d=read(folder/'completed.json')
        size=0
        for filename,item in d['files'].items():
            p=folder/filename; expected=prepared['authenticated_inputs'][str(p)]; dig=item if type(item) is str else item['sha256']
            bind(p,dig,expected['bytes']); size+=expected['bytes']
        result[name]={'completed_sha256':ident['sha256'],'recorded_wall_seconds':d['wall_seconds'],'wall_scope':d.get('wall_scope','Whole preparation wall as recorded by the inherited producer; no retrospective scope expansion'),'recorded_manifest_payload_bytes':size,'completion_receipt_bytes':ident['bytes'],'interpretation':'Prior shared cache preparation, charged separately; not rerun or allocated per arm'}
    return result


def write(p,x):
    with p.open('x') as f: json.dump(x,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--completed-sha256'); parser.add_argument('--report-summary-sha256'); parser.add_argument('--report-receipt-sha256')
    parser.add_argument('--failure-only',action='store_true'); parser.add_argument('--failed-sha256')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); source=sha(__file__)
    try:
        plan,prepared=freeze_identity()
        if args.failure_only:
            need(args.failed_sha256 is not None and not (RUN/'completed.json').exists(),'Failure-only terminal boundary')
            bind(RUN/'failed.json',args.failed_sha256); fail=read(RUN/'failed.json')
            need(fail['status']=='failed' and fail['version']=='dialogue-conditional-training-v1','Failure receipt')
            # Opaque hash every retained file, never load prediction tensors or compute quality.
            for p in RUN.rglob('*'):
                if p.is_file(): bind(p,sha(p),p.stat().st_size)
            result={'status':'completed','audit_scope':'failure-only inventory; no quality metrics computed','producer_status':'failed','producer_failed_sha256':args.failed_sha256,'plan_sha256':PLAN_SHA,'failure':fail,'retained_files':{k:v for k,v in INPUTS.items() if k.startswith('runs/dialogue-conditional-v1/training-01/')},'limits':['File inventory and terminal metadata authenticated. No independent completed-fit/partial numerical replay.']}
        else:
            need(all([args.completed_sha256,args.report_summary_sha256,args.report_receipt_sha256]),'Three external completion/report pins required')
            done,size=authenticate_done(args.completed_sha256,plan); rows=metadata(plan)
            recs,counts=verify_training(done,plan,rows)
            # No prediction or task-quality reads occur until complete technical membership passes above.
            bind(REPORT/'summary.json',args.report_summary_sha256); bind(REPORT/'receipt.json',args.report_receipt_sha256)
            report,receipt=read(REPORT/'summary.json'),read(REPORT/'receipt.json')
            need(receipt['status']=='completed' and receipt['source_run_completed_sha256']==args.completed_sha256 and receipt['summary_sha256']==args.report_summary_sha256 and receipt['reporter_sha256']==plan['source_sha256']['scripts/report_dialogue_conditional.py'],'Reporter receipt linkage')
            fits,refs,primary=analyze(rows['dev'],recs,plan,prepared)
            compare(report['fits'],fits,'fits'); compare(report['references'],refs,'references'); compare(report['primary'],primary,'primary')
            need(report['technical_admission_passed'] is True and receipt['technical_admission_passed'] is True and report['continuation_allowed']==receipt['continuation_allowed']==primary['primary_nll_rule_passed'] and report['architecture_advantage_established'] is False,'Technical and scientific decision agreement')
            need(report['source_run_completed_sha256']==args.completed_sha256 and report['plan_sha256']==PLAN_SHA,'Report header identities')
            compare(report['cost']['fits'],{name:{k:r[k] for k in ('wall_seconds','counts','configuration','training_work','evaluation')} for name,r in recs.items()},'reported per-fit costs')
            compare(report['cost']['inherited_cache_preparation'],inherited_costs(prepared),'inherited costs')
            compare(report['cost']['conditional_metadata_preparation'],{'recorded_wall_seconds':prepared['wall_seconds'],'wall_scope':prepared['wall_scope'],'manifest_payload_bytes':sum(v['bytes'] for v in prepared['files'].values())},'metadata costs')
            need(report['cost']['whole_run_wall_seconds']==done['wall_seconds'] and report['cost']['process_lifetime_peak_rss_bytes']==done['process_lifetime_peak_rss_bytes'] and report['cost']['output_bytes']==size,'Reported whole costs')
            result={'status':'completed','audit_scope':'independent saved-output arithmetic and technical closure','plan_sha256':PLAN_SHA,'execution_completed_sha256':args.completed_sha256,'source_files_verified':18,'execution_files_verified':43,'all_reported_metrics_and_decisions_agree':True,'primary':primary,'fits':fits,'references':refs,'counts':counts,'cost':{'whole_run_wall_seconds':done['wall_seconds'],'process_lifetime_peak_rss_bytes':done['process_lifetime_peak_rss_bytes'],'output_bytes':size},'continuation_allowed':primary['primary_nll_rule_passed'],'scope_limits':['No experiment/report/model modules imported; no model, encoder, optimizer, checkpoint deserialization or new RNG calls.','Mean/Brier/NLL arithmetic independently recomputed with float64 and math.fsum. Reporter comparison tolerance 1e-12 is arithmetic-only; decision signs use no epsilon.','All prepared labels/address semantics inherited from the separately authenticated independent preparation audit, with canonical joins and adjacent prior IDs rechecked.','Opaque checkpoint bytes, initial tensor digests, gradients, optimizer returns, forward invariant checks, actual clocks/RSS/runtime are authenticated execution witnesses, not independently replayed.','Saved epoch permutations checked complete and byte-identical to published freeze; RNG generation recipe is source-bound, not redrawn.','Raw prediction matrices and per-row labels remain private. Previous-gold inputs and exposed development cohort prevent rollout/deployment/architecture claims.','Local Git bytes and ancestry checked; remote publication timing is not independently attested.']}
        for name,item in INPUTS.items():
            p=ROOT/name; need(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'End artifact stability '+name)
        write(args.out/'summary.json',result)
        receipt={'status':'completed','source_path':str(Path(__file__).resolve()),'source_sha256':source,'files':{'summary.json':{'sha256':sha(args.out/'summary.json'),'bytes':(args.out/'summary.json').stat().st_size}},'plan_sha256':PLAN_SHA,'published_commit':COMMIT,'authenticated_inputs':INPUTS,'audit_wall_seconds':time.monotonic()-started,'new_model_calls':0,'new_encoder_calls':0,'new_optimizer_steps':0,'new_rng_calls':0}
        write(args.out/'receipt.json',receipt)
        print(json.dumps({'status':'completed','summary_sha256':sha(args.out/'summary.json'),'receipt_sha256':sha(args.out/'receipt.json'),'source_sha256':source}))
    except BaseException as e:
        try: write(args.out/'failed.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'source_sha256':source,'plan_sha256':PLAN_SHA,'wall_seconds':time.monotonic()-started})
        except BaseException as secondary:
            if callable(getattr(e,'add_note',None)): e.add_note('Audit failure preservation error: '+repr(secondary))
        raise

if __name__=='__main__': main()
