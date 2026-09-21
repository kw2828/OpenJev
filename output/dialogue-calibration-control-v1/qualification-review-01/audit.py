"""Saved-only qualification review, independently recomputing parity/projection.

No neural imports or execution. Shared authentication checks byte/source/runtime
and parent lifecycle; restoration and hardware counters remain producer witnesses.
Original full NPZ arrays must be decompressed, but arithmetic uses only the two
predeclared dialogues, with no targets, task scores, fitting or gate changes.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
PLAN=ROOT/'output/dialogue-calibration-control-v1/plan.json'
PLAN_PIN='e706e40cd4a0d95fb1536ce105074c071b57a6bc4652ec57b9243672542a413c'
PREP=ROOT/'runs/dialogue-calibration-control-v1/preparation-01'
PREP_PIN='896099f38f7316f23e549d1271d81f90a2884028e6c0281dcf3c035b6ee286e9'
PREP_END=ROOT/'output/dialogue-calibration-control-v1/preparation-process-01.terminal.json'
PREP_END_PIN='9a3f54de5ee09259fdfafa9e569556fc51ff78c3bf7ea8b29842e9944aa05bd3'
RUN=ROOT/'runs/dialogue-calibration-control-v1/qualification-01'
RUN_PIN='ca06666f1849a49c3e230f8a1fa6f485f5214dbdbe36c9f7fccde1a2cf2e2a94'
END=ROOT/'output/dialogue-calibration-control-v1/qualification-process-01.terminal.json'
END_PIN='b9ef0d20799f1e8a1fda7b565d57378a9bac7d7b3c53de5a48425d9bb6cf644b'
COMMON_PIN='452016fcdb611199fa4fafcbb54eeb4ab92cdc6f54b94a17d2802eb2223b90dd'
CLOCK_PIN='cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
PREP_AUDIT=ROOT/'output/dialogue-calibration-control-v1/preparation-audit-01'
PREP_AUDIT_SOURCE='863e6476dbbd116b78780de18b951179ef22ecdb29fc74da413029fcc36e60da'
PREP_AUDIT_RECEIPT='4a033dfcea9c9ae29a84a8db281aa5b45e482d2cdbbc063a827a8a5218882618'
ENCODER=('input_texts','content_tokens','encoder_sequences','encoder_calls','special_token_positions',
         'valid_token_positions','padded_token_positions','padded_attention_positions','padding_token_positions',
         'overlength_texts_chunked','truncated_tokens')
HOOKS=('encoder_calls','encoder_sequences','valid_token_positions','padded_token_positions','padded_attention_positions')
COUNTS=('forward_calls','forward_returned','advance_calls','advance_returned','valid_turns',
        'executed_valid_question_slots','real_question_updates','incoming_checks','feature_checks','result_checks',
        'mass_checks','mass_above_one_count')
MAXIMA=('incoming_max_sum_error','feature_max_sum_error','result_max_sum_error','mass_max_overshoot')
MEASURES=('encoder_calls','padded_attention_positions','real_question_updates')
ARMS=('frozen_original','frozen_numbers','trainable_original','trainable_numbers')
FIT_ORDER=[f'{arm}-{seed}' for seed in (6901,6902,6903) for arm in ARMS]
LIMITS={'wall_seconds':300,'rss_bytes':4*1024**3,'output_bytes':32*1024**2}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def need(ok,why):
    if not ok:raise ValueError(why)


def rows(p):
    with Path(p).open() as f:return [json.loads(line) for line in f]


def finite_positive(x):return type(x) in (float,int) and math.isfinite(x) and x>0


def close(a,b,why):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),why)


def merge_invariants(records):
    return {**{k:sum(r[k] for r in records) for k in COUNTS},
            **{k:max(r[k] for r in records) for k in MAXIMA},
            'mass_min':min(r['mass_min'] for r in records),'mass_max':max(r['mass_max'] for r in records),'tolerance':2e-6}


def check_invariants(inv,work):
    expected={'forward_calls':1,'forward_returned':1,
              **dict.fromkeys(('advance_calls','advance_returned','valid_turns'),work['public_user_turns']),
              **dict.fromkeys(('executed_valid_question_slots','real_question_updates','incoming_checks',
                              'feature_checks','result_checks','mass_checks'),work['real_question_updates'])}
    need(set(inv)==set(COUNTS)|set(MAXIMA)|{'mass_min','mass_max','tolerance'},'Exact invariant fields')
    need(all(type(inv[k]) is int and inv[k]==v for k,v in expected.items()),'Every declared state update')
    need(inv['tolerance']==2e-6 and type(inv['mass_above_one_count']) is int
         and 0<=inv['mass_above_one_count']<=work['real_question_updates'],'Mass counts/tolerance')
    need(all(math.isfinite(inv[k]) and 0<=inv[k]<=2e-6 for k in MAXIMA),'Normalization witnesses')
    need(0<=inv['mass_min']<=inv['mass_max']<=1+2e-6,'Mass bounds')


def array_parity(actual,old,mask,np):
    for x in (actual,old):
        need(x.dtype==np.float32 and x.shape==mask.shape and np.isfinite(x[mask]).all()
             and np.isneginf(x[~mask]).all(),'Saved supported f32/padding')
    a,b=actual.astype(np.float64),old.astype(np.float64)
    mass=lambda x:max(abs(math.fsum(float(v) for v in np.exp(row))-1) for row in x)
    actual_mass,old_mass=mass(a),mass(b)
    log_error=float(np.max(np.abs(a[mask]-b[mask])))
    probability_error=float(np.max(np.abs(np.exp(a)-np.exp(b))))
    choices=bool(np.array_equal(np.argmax(a,axis=1),np.argmax(b,axis=1)))
    need(actual_mass<=2e-6 and old_mass<=2e-6 and log_error<=1e-5 and probability_error<=1e-6 and choices,'Exact fixed parity gate')
    return {'rows':len(a),'maximum_supported_log_error':log_error,'maximum_probability_error':probability_error,
            'canonical_first_argmax_equal':choices,'maximum_raw_mass_error':actual_mass,'baseline_maximum_raw_mass_error':old_mass}


def inspect(ctx,done,prep,prep_end,common,np,check):
    need(done['fit_order']==FIT_ORDER==ctx['science']['fit_order'] and len(done['fits'])==12,'All12 fixed final checkpoints')
    need(done['prepared_sha256']==PREP_PIN and done['prepared_terminal_sha256']==PREP_END_PIN
         and done['parity_passed'] is True and done['completed_forwards']==24,'Preparation/forward joins')
    need(done['model_weight_updates']==0 and done['optimizer_created'] is False and done['temperature_applied'] is False
         and done['official_test_opened'] is False and done['task_metrics_computed'] is False,'Qualification-only scope')
    need(not (ROOT/'runs/dialogue-calibration-control-v1/inference-01').exists(),'No downstream inference attempt')
    need(sha(PREP_AUDIT/'audit.py')==PREP_AUDIT_SOURCE and sha(PREP_AUDIT/'receipt.json')==PREP_AUDIT_RECEIPT,'Independent preparation source/receipt')
    inherited=read(PREP_AUDIT/'receipt.json')
    need(inherited['agreement'] is True and inherited['preparation_completed_sha256']==PREP_PIN,'Preparation audit identity')
    for name,record in inherited['files'].items():
        need(sha(PREP_AUDIT/name)==record['sha256'] and (PREP_AUDIT/name).stat().st_size==record['bytes'],'Preparation audit bytes')
    # Only its previously independent pure geometry formula is reused, never producer math.
    spec=importlib.util.spec_from_file_location('calibration_preparation_geometry_audit',PREP_AUDIT/'audit.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cases=read(PREP/'replay-cases.json')['cases']
    all_old=rows(ctx['original_prepared']/'actors-dev.jsonl')
    actor_index={a['dialogue_id']:a for a in all_old}
    need(len(actor_index)==2363 and len(cases)==2,'Complete original DEV and two replay cases')
    first=all_old[0]['dialogue_id']
    old_work={did:helper.work(a) for did,a in actor_index.items()}
    largest=min(set(actor_index)-{first},key=lambda did:(-old_work[did]['padded_attention_positions'],did))
    need([c['dialogue_id'] for c in cases]==[first,largest],'Independent fixed replay selection')
    old_rows=rows(ctx['prior'].run/'evaluation-rows.jsonl')
    need(len(old_rows)==62329 and all(r['row_index']==i for i,r in enumerate(old_rows)),'Old row order')
    fields=('row_index','source_row_index','time','turn_index','query_position','query_index','query_id',
            'service','slot','candidate_ids','candidate_values')
    for case in cases:
        did=case['dialogue_id'];actor=actor_index[did]
        refs=[{k:r[k] for k in fields} for r in old_rows if r['dialogue_id']==did]
        need(case['work']==old_work[did] and case['endpoints']==refs and case['row_indices']==[r['row_index'] for r in refs],'Complete public replay references')
        for r in refs:
            j,t=r['query_position'],r['time']
            need(r['query_index']==actor['query_ids'][j] and r['turn_index']==actor['user_turn_indices'][t]
                 and r['candidate_ids']==actor['candidate_ids'][j],'Actor time/schema/candidate references')
    calibration=rows(PREP/'actors.jsonl')
    work={a['dialogue_id']:helper.work(a) for a in calibration}
    selected=read(PREP/'selection.json')['selected_ids']
    need(len(work)==512 and [a['dialogue_id'] for a in calibration]==selected,'Complete calibration work')
    totals={k:sum(w[k] for w in work.values()) for k in MEASURES}
    prepared_totals=read(PREP/'workloads.json')['all']['totals']
    need(all(totals[k]==prepared_totals[k] for k in MEASURES),'Independent full-cohort projection denominators')
    retained,loads,all_cases=[],[],[]
    for fit_id,published in zip(FIT_ORDER,done['fits'],strict=True):
        check();directory=RUN/fit_id;fit=read(directory/'completed.json');arm,seed=fit_id.rsplit('-',1)
        need(published=={**fit,'completed_sha256':sha(directory/'completed.json')},'Exact root/fit receipt join')
        need(fit['status']=='completed' and fit['fit_id']==fit_id and fit['arm']==arm and fit['seed']==int(seed)
             and fit['endpoint_rows']==sum(len(c['endpoints']) for c in cases),'Fit identity/rows')
        need(set(fit['files'])=={'cases.jsonl','case-00.npz','case-01.npz'},'Fit payload closure')
        common.manifest(directory,fit['files'])
        oldfit=read(ctx['prior'].run/fit_id/'completed.json');witness=fit['witness']
        need(oldfit['status']=='completed' and oldfit['fit_id']==fit_id and oldfit['arm']==arm and oldfit['seed']==int(seed),'Original checkpoint identity')
        need(witness['checkpoint_sha256']==oldfit['checkpoint']['sha256']==oldfit['files']['weights.pt']['sha256'], 'Pinned final checkpoint byte identity')
        need(witness['arm']==arm and witness['version']=='dialogue-calibration-inference-v1'
             and witness['restored_sha256']==witness['final_sha256']
             and set(witness['restored_sha256'])=={'memory','encoder'}
             and witness['restored_sha256']['encoder']==oldfit['final_encoder_sha256'], 'Restored unchanged parameter witnesses')
        need(witness['encoder_device']=='mps:0' and witness['memory_device']=='cpu' and witness['dtype']=='float32'
             and witness['chunk_tokens']==254 and witness['encoder_batch']==32 and witness['synthetic_injection'] is False
             and witness['optimizer_created'] is False and witness['temperature_applied'] is False,'Qualified inference route')
        expected_counts={p+'_'+suffix:(1 if p in ('checkpoint_load','encoder_load') else sum(c['work']['encoder_calls'] for c in cases)
                                     if p=='encoder_forward' else 2)
                         for p in ('checkpoint_load','encoder_load','encoder_forward','public_forward','encoding','memory_forward')
                         for suffix in ('attempts','returns')}
        need(witness['counts']==expected_counts,'All actual attempt/return counts')
        journal=rows(directory/'cases.jsonl')
        need(len(journal)==len(fit['cases'])==2,'Two journal/case records')
        with np.load(ctx['prior'].run/fit_id/'predictions.npz',allow_pickle=False) as archive:
            need(set(archive.files)=={'row_indices','log_probs'},'Original prediction artifact keys')
            indices=archive['row_indices'];baseline=archive['log_probs']
            need(indices.dtype==np.int64 and np.array_equal(indices,np.arange(62329))
                 and baseline.dtype==np.float32 and baseline.shape==(62329,12),'Canonical original raw distribution geometry')
        parity_records=[]
        for ci,(record,case,event) in enumerate(zip(fit['cases'],cases,journal,strict=True)):
            check()
            need(event=={k:v for k,v in record.items() if k!='case_seconds'},'Exact journal/case join except paid elapsed')
            need(record['case_index']==ci and record['dialogue_id']==case['dialogue_id']
                 and record['row_indices']==case['row_indices'] and record['work']==case['work'],'Case identity/work')
            need(finite_positive(record['payload_seconds']) and finite_positive(record['case_seconds'])
                 and record['case_seconds']>=record['payload_seconds'],'Paid case includes journal/storage/sync')
            need(record['encoder_work']=={k:case['work'][k] for k in ENCODER},'All encoder logical work')
            check_invariants(record['invariants'],case['work'])
            with np.load(directory/f'case-{ci:02}.npz',allow_pickle=False) as archive:
                need(set(archive.files)=={'row_indices','log_probs'},'Replay artifact keys')
                ids=archive['row_indices'];actual=archive['log_probs']
                need(ids.dtype==np.int64 and ids.tolist()==case['row_indices'],'Replay row index identity')
            support=np.arange(12)[None,:]<np.array([len(r['candidate_ids']) for r in case['endpoints']])[:,None]
            numerical=array_parity(actual,baseline[ids],support,np)
            need(set(numerical)==set(record['parity']),'Parity record schema')
            for key,value in numerical.items():
                if type(value) in (int,bool):need(value==record['parity'][key],'Parity discrete witness')
                else:close(value,record['parity'][key],'Parity numeric witness '+key)
            parity_records.append({'case_index':ci,'dialogue_id':case['dialogue_id'],**numerical,
                                   'case_seconds':record['case_seconds'],'payload_seconds':record['payload_seconds']})
            all_cases.append({'fit_id':fit_id,**record})
        del baseline
        aggregate={k:sum(c['work'][k] for c in cases) for k in ENCODER}
        need(witness['encoder_work']==aggregate and witness['encoder_attempted_work']==witness['encoder_returned_work']
             =={k:aggregate[k] for k in HOOKS},'Complete fit encoder count arithmetic')
        need(witness['invariants']==merge_invariants([c['invariants'] for c in fit['cases']]),'State invariant aggregate')
        need(finite_positive(fit['loading_seconds']) and finite_positive(fit['wall_seconds'])
             and fit['wall_seconds']>=fit['loading_seconds']+sum(c['case_seconds'] for c in fit['cases']),'Complete fit timing inclusion')
        loads.append(fit['loading_seconds'])
        retained.append({'fit_id':fit_id,'loading_seconds':fit['loading_seconds'],'wall_seconds':fit['wall_seconds'],
                         'checkpoint_sha256':witness['checkpoint_sha256'],'cases':parity_records})
    estimates,dominators={},{}
    for key in MEASURES:
        top=max(all_cases,key=lambda c:c['case_seconds']/c['work'][key]);rate=top['case_seconds']/top['work'][key]
        count=12*totals[key]
        estimates[key]={'maximum_seconds_per_work_unit':rate,'all_fit_work':count,'projected_seconds':rate*count}
        dominators[key]={'fit_id':top['fit_id'],'case_index':top['case_index'],'dialogue_id':top['dialogue_id'],
                         'case_seconds':top['case_seconds'],'work':top['work'][key]}
    variable=max(x['projected_seconds'] for x in estimates.values());loading=math.fsum(loads)
    total=variable+loading+prep_end['wall_seconds']
    projection={'measures':estimates,'variable_seconds':variable,'loading_seconds':loading,
                'preparation_seconds':prep_end['wall_seconds'],'total_seconds':total,'threshold_seconds':1800,
                'admitted':total<=1800,
                'scope':'Heuristic screening of 12 x 512 forwards, not a runtime guarantee; fixed hard cap remains.'}
    need(projection==read(RUN/'projection.json')==done['projection'],'Exact fixed projection and cost decision')
    need(projection['admitted'] is False,'Preserve completed failed admission')
    need(sum(f['wall_seconds'] for f in retained)<=done['wall_seconds'],'Fit times contained in whole phase')
    return {'status':'completed','agreement':True,'parity_passed':True,'cost_admitted':False,
            'completed_fits':12,'completed_replay_forwards':24,'compared_endpoint_rows':sum(c['parity']['rows'] for c in all_cases),
            'actual_encoder_calls_recorded':sum(c['work']['encoder_calls'] for c in all_cases),
            'state_question_updates_recorded':sum(c['work']['real_question_updates'] for c in all_cases),
            'maximum_log_error':max(c['parity']['maximum_supported_log_error'] for c in all_cases),
            'maximum_probability_error':max(c['parity']['maximum_probability_error'] for c in all_cases),
            'cases':retained,'projection':projection,'dominating_measure':max(estimates,key=lambda k:estimates[k]['projected_seconds']),
            'dominating_cases':dominators,'calibration_work_one_visit':totals,
            'interpretation':'All declared saved endpoints replay exactly within tolerance. The unchanged conservative cost rule fails; no full inference or temperature fit authorized.',
            'limits_of_inference':'Timing and actual hardware/restore counters are authenticated producer witnesses, not new measurements. Matching saved endpoints does not establish unseen hidden-state equality. No causal timing attribution or retiming.'}


def main():
    need(not (OUT/'started.json').exists(),'Exclusive invocation')
    source_pin=sha(__file__);clock=deadline=None;elapsed=None;handler=None;progress={'stage':'start'}
    try:
        need(sha(ROOT/'scripts/dialogue_calibration_common.py')==COMMON_PIN,'Common source pin')
        need(sha(ROOT/'src/openjev/research/suspend_clock.py')==CLOCK_PIN,'Clock source pin')
        from openjev.research.suspend_clock import SuspendClock
        clock=SuspendClock();deadline=clock.deadline_after(300)
        import dialogue_calibration_common as common
        import numpy as np
        def check():
            nonlocal elapsed
            elapsed=clock.now_ns()-deadline.started_ns
            need(elapsed<300*10**9 and common.peak_rss()<=LIMITS['rss_bytes'],'Native deadline/RSS cap')
            need(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=LIMITS['output_bytes'],'Audit output cap')
        def alarm(*_):raise TimeoutError('Audit wall cap')
        handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,300)
        write(OUT/'started.json',{'status':'started','source_sha256':source_pin,'plan_sha256':PLAN_PIN,
              'qualification_completed_sha256':RUN_PIN,'qualification_terminal_sha256':END_PIN,'limits':LIMITS,
              'clock_backend':clock.backend,'started_ns':deadline.started_ns,'deadline_ns':deadline.expires_ns,
              'npz_opened_before_freeze':False,'experimental_model_calls':0})
        ctx=common.authenticate(SimpleNamespace(plan=PLAN,plan_sha256=PLAN_PIN,out=RUN,command='qualify'),
                                SimpleNamespace(check=check,progress=progress))
        prep=common.authenticate_output(PREP,PREP_PIN,ctx,'prepare')
        prep_end=common.authenticate_terminal(PREP_END,PREP_END_PIN,prep)
        done=common.authenticate_output(RUN,RUN_PIN,ctx,'qualify')
        terminal=common.authenticate_terminal(END,END_PIN,done)
        progress['stage']='saved_array_review'
        summary=inspect(ctx,done,prep,prep_end,common,np,check)
        common.bind(ctx['plan']['sources'],check);common.bind(ctx['science']['source_sha256'],check)
        common.authenticate_output(RUN,RUN_PIN,ctx,'qualify')
        need(sha(__file__)==source_pin and sha(PLAN)==PLAN_PIN and sha(END)==END_PIN,'Source/input stability')
        check();write(OUT/'summary.json',summary);check()
        receipt={'status':'completed','agreement':True,'source_sha256':source_pin,'plan_sha256':PLAN_PIN,
                 'qualification_completed_sha256':RUN_PIN,'qualification_terminal_sha256':END_PIN,
                 'preparation_completed_sha256':PREP_PIN,'preparation_terminal_sha256':PREP_END_PIN,
                 'preparation_audit_receipt_sha256':PREP_AUDIT_RECEIPT,'preparation_geometry_source_sha256':PREP_AUDIT_SOURCE,
                 'parent_returncode':terminal['returncode'],'parent_group_absent':terminal['group_absent'],
                 'parent_wall_seconds':terminal['wall_seconds'],'parity_passed':True,'cost_admitted':False,
                 'experimental_model_calls':0,'tokenizer_calls':0,'checkpoint_tensors_loaded':0,
                 'official_test_opened':False,'task_quality_scored':False,'temperature_fits':0,
                 'source_count_new':len(ctx['plan']['sources']),'source_count_inherited':len(ctx['science']['source_sha256']),
                 'scope':'Common byte/lifecycle authentication reused. Independent parity of all24 saved cases, canonical actor/row/checkpoint metadata and exact projection. Pure geometry from pinned independent preparation audit.',
                 'limits':LIMITS,'clock_backend':clock.backend,'started_ns':deadline.started_ns,'deadline_ns':deadline.expires_ns,
                 'elapsed_ns':elapsed,'wall_seconds':elapsed/1e9,'peak_rss_bytes':common.peak_rss(),
                 'timing_scope':'Native start through pre-receipt check; publication rechecked before return',
                 'files':{n:{'bytes':(OUT/n).stat().st_size,'sha256':sha(OUT/n)} for n in ('audit.py','started.json','summary.json')}}
        write(OUT/'receipt.json',receipt);check()
        print(json.dumps({'status':'completed','agreement':True,'parity_passed':True,'cost_admitted':False,
                          'projected_seconds':summary['projection']['total_seconds'],'receipt_sha256':sha(OUT/'receipt.json'),
                          'summary_sha256':sha(OUT/'summary.json'),'wall_seconds':elapsed/1e9}))
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        if (OUT/'receipt.json').exists():(OUT/'receipt.json').rename(OUT/'invalid-receipt.json')
        try:
            write(OUT/'failed.json',{'status':'failed','source_sha256':source_pin,'error_type':type(error).__name__,
                  'error':str(error),'progress':progress,'last_successful_elapsed_ns':elapsed,'experimental_model_calls':0})
        except BaseException as cleanup:error.add_note('Failure receipt error: '+repr(cleanup))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if handler is not None:signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':main()
