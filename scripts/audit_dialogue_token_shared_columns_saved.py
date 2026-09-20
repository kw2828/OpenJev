from pathlib import Path
import argparse, collections, hashlib, importlib.metadata, json, math, platform, statistics, subprocess
ROOT=Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
RUN=ROOT/'output/dialogue-token-shared-columns-v1/qualification-01'
FREEZE=ROOT/'output/dialogue-token-shared-columns-v1/protocol-01'
PARENT=ROOT/'output/dialogue-token-required-fill-v1/qualification-01'
OUT=ROOT/'output/dialogue-token-shared-columns-v1/audit-01'
COMPLETE=None  # Required external CLI pin, supplied only after terminal execution.
PLAN='30503853fd8c9d2da2719136f0ab361e1ce52b62031b6a0085d4e2ac54af8650'
COMMIT='3e58d2f7384327d157d29c895ff76e7391db2adc'
PCOMPLETE='465e6ec7a6326c063582262529522f7ef66d0449327363fc3e7ce9def46b3665'

def audit():
 global normmax,overshoots
 checks=collections.Counter(); inputs={}
 def req(ok,label):
  if not ok: raise ValueError(label)
  checks[label.split(':')[0]]+=1

 def read(p): return json.loads(p.read_text())
 def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
 def bind(p,h):
  req(p.is_file() and not p.is_symlink() and digest(p)==h,'sha256: '+str(p)); inputs[str(p.relative_to(ROOT))]={'sha256':h,'bytes':p.stat().st_size}
 def finite(x): return type(x) in (float,int) and math.isfinite(x)
 def near(a,b,label): req(finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label)
 def tree(p): return {str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()}
 def manifest(p,pin,n):
  bind(p/'completed.json',pin); d=read(p/'completed.json');req(d['status']=='completed','completion status')
  req(tree(p)==set(d['files'])|{'completed.json'} and len(tree(p))==n,'exact member closure')
  for k,v in d['files'].items():
   req(not Path(k).is_absolute() and '..' not in Path(k).parts,'safe payload path');bind(p/k,v['sha256']);req((p/k).stat().st_size==v['bytes'],'payload bytes')
  return d

 done=manifest(RUN,COMPLETE,92);parent_done=manifest(PARENT,PCOMPLETE,87)
 bind(RUN/'plan.json',PLAN);bind(FREEZE/'plan.json',PLAN)
 plan=read(RUN/'plan.json');summary=read(RUN/'summary.json');started=read(RUN/'started.json');recipe=plan['recipe'];sources=plan['source_sha256']
 req(len(sources)==71 and done['source_sha256']==sources,'71 source closure')
 expected_members={'plan.json','started.json','operations.jsonl','summary.json','completed.json'}|{'sources/'+s for s in sources}|{'cases/'+c['name']+'.json' for c in recipe['cases']}
 req(tree(RUN)==expected_members,'exact source/case namespace')
 source_verified={}
 for name,h in sources.items():
  for p in (ROOT/name,RUN/'sources'/name,FREEZE/'sources'/name):bind(p,h)
  data=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
  req(hashlib.sha256(data).hexdigest()==h,'pre-run Git source identity')
  source_verified[name]=h
 req(hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+str((FREEZE/'plan.json').relative_to(ROOT))],cwd=ROOT)).hexdigest()==PLAN,'pre-run Git plan identity')
 freeze=read(FREEZE/'completed.json')
 req(freeze['status']=='completed' and freeze['phase']=='freeze' and freeze['plan_sha256']==PLAN and freeze['source_sha256']==sources and freeze['model_calls']==0 and freeze['no_retry'] is True,'freeze receipt')
 bind(FREEZE/'completed.json',digest(FREEZE/'completed.json'))
 req(not any((FREEZE/x).exists() for x in ('failed.json','late-completion.json')),'no failed freeze')
 current=plan;lineage=[]
 for ancestor,count in [('required_fill',66),('consolidation',61),('factoring',56),('projection',51),('packing',46),('batching',40),('training',35)]:
  pp=FREEZE/(ancestor.replace('_','-')+'-plan.json');bind(pp,current[ancestor+'_plan_sha256']);p=read(pp)
  req(len(p['source_sha256'])==count and all(sources[k]==v for k,v in p['source_sha256'].items()),'ancestor source closure')
  req(p['runtime']==plan['runtime'] and p['loss_counts']==plan['loss_counts'] and p['loss_weights']==plan['loss_weights'],'ancestor runtime/loss identity')
  if ancestor!='training':
   cp=FREEZE/(ancestor.replace('_','-')+'-completed.json');bind(cp,current[ancestor+'_completed_sha256']);cd=read(cp)
   req(cd['status']=='completed' and cd['phase']=='freeze' and cd['plan_sha256']==digest(pp) and cd['source_sha256']==p['source_sha256'] and cd['model_calls']==0 and cd['no_retry'] is True,'ancestor freeze receipt')
  lineage.append({'name':ancestor,'plan_sha256':digest(pp),'sources':count}); current=p
 for k in ['learning_rate','weight_decay','gradient_clip','threads','projection_dim','hidden_dim','gru_width']:req(current['config'][k]==recipe[k],'original recipe inherited')
 req(plan['loss_weights']==[sum(plan['loss_counts'].values())/(3*plan['loss_counts'][str(i)]) for i in range(3)],'weighted loss arithmetic')
 rt={'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'numpy':importlib.metadata.version('numpy'),'platform':platform.platform()}
 req(plan['runtime']==done['runtime']==rt,'runtime metadata identity')
 req(started['phase']=='run' and started['no_retry'] is True and started['request']['plan_sha256']==PLAN and started['request']['plan']=='output/dialogue-token-shared-columns-v1/protocol-01/plan.json' and started['request']['out']=='runs/dialogue-token-shared-columns-v1/qualification-01','started request')
 req(started['source_sha256']==sources['scripts/qualify_dialogue_token_batching.py'] and started['request']['shared_columns_harness_sha256']==sources['scripts/qualify_dialogue_token_shared_columns.py'],'started source identity')
 req(done['version']==plan['version']=='dialogue-token-shared-columns-qualification-v1' and done['plan_sha256']==PLAN and done['no_retry'] is True and done['corpus_reads']==done['encoder_calls']==0 and done['full_training_authorized'] is False,'execution scopes')
 base={'device':'cpu','dtype':'float32','deterministic':True,'threads':4,'interop_threads':1,'input_dim':384,'projection_dim':64,'hidden_dim':64,'gru_width':16,'parameters':173186,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,'measured_pairs':4,'warm_updates_per_path':1,'minimum_speed_ratio':.9,'other_speed_ratio':1.1,'maximum_rss_bytes':6*1024**3,'whole_cap_seconds':300.,'operation_records':192,'optimizer_updates':160,'parity_forward_backward_passes':32,'output_atol':1e-5,'output_rtol':1e-4,'gradient_atol':1e-5,'gradient_rtol':1e-4,'loss_atol':1e-6,'loss_rtol':1e-5,'packing_bucket_width':16,'projection_width':64}
 for k,v in base.items():req(recipe[k]==v,'fixed recipe '+k)
 paths=['original','shared_columns'];orders=[paths,paths[::-1]]*2;req(recipe['pair_orders']==orders,'pair order recipe')
 cases=[]
 for group,shape,startseed in [('minimum',[1,6,1,7,53],91201),('medium',[32,15,10,12,80],91205),('large',[32,23,10,12,90],91209),('geometry',[32,23,10,12,90],91301)]:
  for j,(mode,head) in enumerate([('slot','readout'),('slot','scalar'),('candidate','readout'),('candidate','scalar')]):cases.append(dict(index=len(cases),name=f'{group}-{mode}-{head}',seed=startseed+j,shape=shape,mode=mode,head=head))
 req(recipe['cases']==cases,'exact sixteen fixed cases')
 geom_name='output/dialogue-token-packing-v1/geometry-01.json';geom=read(RUN/'sources'/geom_name)
 req(plan['geometry_sha256']==sources[geom_name]==recipe['geometry_sha256'],'geometry pin')
 ref_fields=['scatter_evidence_scalars','dense_evidence_scalars','dense_turn_projection_positions','projected_scatter_scalars','dense_projected_scalars','projected_bias_fill_positions','projected_skipped_positions','projected_scatter_bytes_float32','dense_projected_bytes_float32','dense_feature_macs','full_fill_reference_positions','full_fill_reference_calls','full_fill_reference_exogenous_scalars','full_fill_reference_hidden_scalars','full_fill_reference_macs','state_column_reference_expressions']
 widths={'raw_evidence':384,'attention':64,'projected_evidence':64,'hidden_preactivation':64,'exogenous_feature':394,'state_feature':2}
 def counts(c):
  B,T,Q,C,L=c['shape'];dialogs=[];cursor=0
  for i in range(B):
   if c['index']>=12:d=geom['dialogs'][i]
   else:
    nt=max(1,T-i%4);nq=max(1,Q-i%3);d={'turn_token_lengths':[max(1,L-(cursor+t)%13) for t in range(nt)],'candidate_counts':[max(3,C-q%3) for q in range(nq)]};cursor+=nt
   dialogs.append(d)
  req([B,max(len(d['turn_token_lengths']) for d in dialogs),max(len(d['candidate_counts']) for d in dialogs),max(max(d['candidate_counts']) for d in dialogs),max(max(d['turn_token_lengths']) for d in dialogs)]==c['shape'],'mask maximum shape')
  turns=sum(len(d['turn_token_lengths']) for d in dialogs);tokens=sum(sum(d['turn_token_lengths']) for d in dialogs)
  realq=sum(len(d['turn_token_lengths'])*len(d['candidate_counts']) for d in dialogs);realc=sum(len(d['turn_token_lengths'])*sum(d['candidate_counts']) for d in dialogs)
  supports=[sum(d['candidate_counts'])+Q-len(d['candidate_counts']) for d in dialogs]
  S=sum(len(d['turn_token_lengths'])*s for d,s in zip(dialogs,supports));N=B*T*Q*C;K=B*Q*C
  group=set();bucket=score=0
  for d,s in zip(dialogs,supports):
   for n in d['turn_token_lengths']:
    k=min(L,16*((n+15)//16));group.add((k,s));bucket+=k;score+=k*s
  w=dict(real_turns=turns,supported_schema_pairs=sum(supports),packed_token_key_positions=tokens,dense_token_key_positions=B*T*L,packed_score_positions=score,dense_score_positions=N*L,packed_evidence_positions=S,dense_evidence_positions=N,pooling_groups=len(group),bucket_token_positions=bucket,packed_raw_token_scalars=tokens*384,grouped_raw_token_scalars=bucket*384,grouped_key_scalars=bucket*64,grouped_schema_scalars=S*64,scatter_evidence_scalars=S*384,dense_evidence_scalars=N*384,packed_turn_projection_positions=S,dense_turn_projection_positions=N,turn_projection_calls=1,empty_turn_projection_calls=int(S==0),projected_scatter_scalars=S*64,dense_projected_scalars=N*64,projected_bias_fill_positions=N,projected_skipped_positions=N-S,projected_scatter_bytes_float32=4*S*64,dense_projected_bytes_float32=4*N*64,exogenous_width=394,group_feature_positions=S,fill_feature_positions=K,state_feature_positions=N,group_feature_calls=int(S>0),fill_feature_calls=1,state_feature_calls=T,group_exogenous_input_scalars=S*394,fill_exogenous_input_scalars=K*394,state_feature_input_scalars=2*N,grouped_head_schema_scalars=2*S*64,grouped_lexical_scalars=10*S,scatter_hidden_scalars=S*64,dense_hidden_scalars=N*64,fill_hidden_scalars=K*64,group_feature_macs=S*394*64,fill_feature_macs=K*394*64,state_feature_macs=2*N*64,factored_feature_macs=(S+K)*394*64+2*N*64,dense_feature_macs=N*396*64,scatter_hidden_bytes_float32=4*S*64,dense_hidden_bytes_float32=4*N*64,pooled_concat_scalars=S*384,pooled_concat_bytes_float32=4*S*384)
  F=sum(Q*C if len(d['turn_token_lengths'])<T else Q*C-support for d,support in zip(dialogs,supports))
  partial=0<F<K
  w.update(fill_feature_positions=F,fill_feature_calls=int(F>0),fill_exogenous_input_scalars=F*394,fill_hidden_scalars=F*64,fill_feature_macs=F*394*64,factored_feature_macs=(S+F)*394*64+2*N*64,full_fill_reference_positions=K,full_fill_reference_calls=1,full_fill_reference_exogenous_scalars=K*394,full_fill_reference_hidden_scalars=K*64,full_fill_reference_macs=K*394*64,omitted_fill_positions=K-F,fill_mask_reduction_positions=B*T+K,fill_mask_boolean_scalars=2*B+2*K,fill_mask_boolean_bytes=2*B+2*K,fill_count_int64_scalars=1,fill_count_int64_bytes=8,fill_index_int64_scalars=2*F if partial else 0,fill_index_int64_bytes=16*F if partial else 0,fill_gathered_schema_scalars=2*F*64 if partial else 0,fill_zero_lexical_scalars=10*F,fill_zero_storage_scalars=K*64 if F<K else 0,fill_schema_scatter_positions=F if partial else 0,fill_schema_scatter_scalars=F*64 if partial else 0,fill_auxiliary_float32_bytes=4*((2*F*64 if partial else 0)+10*F+(K*64 if F<K else 0)),fill_mask_predicate_calls=1)
  req(F==[0,3025,3025,3789][c['index']//4],'required fill fixed geometry')
  w.update(state_column_view_expressions=1,state_column_reference_expressions=T,state_column_view_scalars=128,state_column_linear_calls=T)
  req(len(w)==74,'74 derived work fields')
  actor=dict(padded_candidate_positions=N,padded_query_positions=B*T*Q,padded_turn_positions=B*T,real_question_steps=realq,real_turns=turns)
  obs=dict(emitted_bytes=B*T*L*384*4,emitted_float32_scalars=B*T*L*384,emitted_token_mask_bytes=B*T*L,emitted_token_prior_bytes=B*T*L*4,evidence_query_projection_positions=K,padded_token_positions=B*T*L,pooling_evidence_positions=N,pooling_score_positions=N*L,raw_cache_prior_bytes_read=tokens*4,raw_cache_token_bytes_read=tokens*384*4,real_candidate_updates=realc,real_public_turns=turns,real_question_updates=realq,schema_candidate_projection_positions=K,schema_query_projection_positions=B*Q,token_key_projection_positions=B*T*L,turn_projection_positions=N,valid_token_positions=tokens)
  sup=sum((ti+qi+i)%4!=3 for i,d in enumerate(dialogs) for ti in range(len(d['turn_token_lengths'])) for qi in range(len(d['candidate_counts'])))
  return w,actor,obs,sup,supports
 normmax=0.;overshoots=0;phase=collections.defaultdict(lambda:collections.defaultdict(float));monitor_totals=collections.Counter()
 def monitor(m,c,actor):
  global normmax,overshoots
  n=actor['real_turns']*c['shape'][2];T=c['shape'][1]
  expected=dict(forward_calls=1,forward_returned=1,advance_calls=T,advance_returned=T,valid_turns=actor['real_turns'],executed_valid_question_slots=n,real_question_updates=actor['real_question_steps'],incoming_checks=n,feature_checks=n,result_checks=n,mass_checks=n if c['head']=='scalar' else 0)
  extra=['incoming_max_sum_error','feature_max_sum_error','result_max_sum_error','mass_max_overshoot','mass_above_one_count','mass_min','mass_max','tolerance']
  req(set(m)==set(expected)|set(extra),'monitor fields')
  for k,v in expected.items():req(type(m[k]) is int and m[k]==v,'monitor coverage '+k);monitor_totals[k]+=v
  req(m['tolerance']==2e-6,'monitor tolerance')
  for k in extra[:4]:req(finite(m[k]) and 0<=m[k]<=2e-6,'normalization witness');normmax=max(normmax,m[k])
  req(type(m['mass_above_one_count']) is int and 0<=m['mass_above_one_count']<=expected['mass_checks'],'mass overshoot count');overshoots+=m['mass_above_one_count']
  if c['head']=='scalar':
   req(finite(m['mass_min']) and finite(m['mass_max']) and 0<=m['mass_min']<=m['mass_max']<=1+2e-6,'mass range');near(m['mass_max_overshoot'],max(0,m['mass_max']-1),'mass overshoot arithmetic');req((m['mass_above_one_count']>0)==(m['mass_max']>1),'mass overshoot consistency')
  else:req(m['mass_min'] is None and m['mass_max'] is None and m['mass_max_overshoot']==0 and m['mass_above_one_count']==0,'readout no mass')
 def packing(p,w):
  req(p['positions_and_scalars']==w and all(type(v) is int for v in p['positions_and_scalars'].values()),'74 actual counters')
  req(p['widths']==widths and p['reference_only_fields']==ref_fields,'width/reference-only semantics')
  req(p['float32_payload_bytes']=={k:4*v for k,v in w.items() if k.endswith('_scalars') and k not in ref_fields and k not in ['state_column_view_scalars','fill_mask_boolean_scalars','fill_count_int64_scalars','fill_index_int64_scalars']},'actual payload scalar bytes')
  req(p['view_only_fields']==['state_column_view_scalars'] and p['logical_view_scalars']=={'state_column_view_scalars':128},'logical view separated from payload')
  req(p['view_scope']=='Shared strided parameter view shape, not copied/allocated payload, backward nodes or memory traffic','view scope')
  req(p['boolean_payload_bytes']=={'fill_mask_boolean_scalars':w['fill_mask_boolean_scalars']},'boolean payload bytes')
  req(p['int64_payload_bytes']=={k:8*w[k] for k in ['fill_count_int64_scalars','fill_index_int64_scalars']},'int64 payload bytes')
  req(p['mac_scope']=='First feature linear layer only; multiply-accumulates, not total model FLOPs or measured time' and p['scope']=='Grouped/pooled-concat/required-fill/state and hidden-scatter logical payloads; mask bool/index int64 charged separately; full fill and earlier dense/projected fields reference-only; view-only fields excluded from payload estimates; aliases are not additive, not traffic or peak memory','cost scope')
 def update(u,c,w,actor,obs,sup,path):
  req(u['actor_shapes']==actor and u['observation_work']==obs and u['supervised_queries']==sup,'update public coverage')
  req(finite(u['seconds']) and u['seconds']>0 and finite(u['loss']) and u['loss']>=0,'update finite values')
  req(set(u['phase_seconds'])=={'assembly','forward_validation_loss','backward_clip','optimizer','scalar_readback'} and all(finite(x) and x>=0 for x in u['phase_seconds'].values()),'phase timing values')
  near(sum(u['phase_seconds'].values()),u['seconds'],'phase timing sum')
  for k,v in u['phase_seconds'].items():phase[path][k]+=v
  if path=='shared_columns':packing(u['packing'],w)
  else:req('packing' not in u,'baseline no optimized counters')
 paritymax=collections.defaultdict(lambda:{'max_abs':0.,'max_tolerance_fraction':0.,'records':0})
 def stat(s,n,neg,kind):
  req(set(s)=={'finite_elements','negative_infinity_elements','max_abs','max_tolerance_fraction'} and s['finite_elements']==n and s['negative_infinity_elements']==neg,'parity finite support '+kind)
  req(finite(s['max_abs']) and s['max_abs']>=0 and finite(s['max_tolerance_fraction']) and 0<=s['max_tolerance_fraction']<=1,'parity bounds witness '+kind)
  for k in ['max_abs','max_tolerance_fraction']:paritymax[kind][k]=max(paritymax[kind][k],s[k])
  paritymax[kind]['records']+=1
 paramcounts={'candidate_projection.bias':64,'candidate_projection.weight':24576,'evidence_query.weight':49152,'feature.0.bias':64,'feature.0.weight':25344,'head.bias':2,'head.weight':128,'query_projection.bias':64,'query_projection.weight':24576,'token_key.weight':24576,'turn_projection.bias':64,'turn_projection.weight':24576}
 req(sum(paramcounts.values())==173186,'parameter schema count')
 operations=[json.loads(x) for x in (RUN/'operations.jsonl').read_text().splitlines()];req(len(operations)==192,'operation count')
 rows=[];cells=[];parent_pairs=[];rss_values=[];case_total=0.;components=collections.Counter()
 for c in cases:
  name=c['name'];row=read(RUN/'cases'/(name+'.json'));oldrow=read(PARENT/'cases'/(name+'.json'));req(row['case']==c==oldrow['case'],'case identity')
  w,actor,obs,sup,supports=counts(c);req(row['expected_packing_work']==w,'expected mask geometry')
  req(row['geometry_sha256']==(plan['geometry_sha256'] if c['index']>=12 else None),'case geometry binding')
  for k in ['sample_sha256','initial_sha256','measured_initial_sha256']:req(row[k]==oldrow[k] and len(row[k])==64,'parent digest identity '+k)
  for k in ['actor_sha256','mask_sha256']:req(row['parity'][k]==oldrow['parity'][k],'parent parity input identity')
  parent_pairs.append({'case':name,**{k:row[k] for k in ['sample_sha256','initial_sha256','measured_initial_sha256']},**{k:row['parity'][k] for k in ['actor_sha256','mask_sha256']}})
  events=operations[c['index']*12:(c['index']+1)*12]
  want=[(name,kind,path,None) for kind in ['parity','warm'] for path in paths]+[(name,'measured',path,i) for i,o in enumerate(orders) for path in o]
  req([(e['case'],e['kind'],e['path'],e.get('pair')) for e in events]==want,'exact event order')
  for e in events:
   monitor(e['monitor'],c,actor)
   if e['path']=='shared_columns':packing(e['packing'],w)
   else:req('packing' not in e,'baseline event no optimized counters')
   if e['kind']=='parity':req(e['forward_backward_completed'] is True and finite(e['loss']) and e['loss']>=0,'parity completed')
  par=row['parity'];req(par['passed'] is True and finite(par['seconds']) and par['seconds']>0,'parity success witness');B,T,Q,C,L=c['shape']
  stat(par['outputs'],T*sum(supports),B*T*Q*C-T*sum(supports),'outputs');stat(par['loss'],1,0,'loss')
  delta=abs(events[0]['loss']-events[1]['loss']);near(par['loss']['max_abs'],delta,'loss parity arithmetic');near(par['loss']['max_tolerance_fraction'],delta/(1e-6+1e-5*abs(events[0]['loss'])),'loss tolerance arithmetic')
  req(set(par['inputs'])==set(map(str,range(8))) and set(par['parameters'])==set(paramcounts),'all gradient keys')
  ns={0:B*T*L*384,2:B*Q*384,3:B*Q*C*384,5:B*T*Q*C*10,7:B*T*L}
  for i in range(8):
   if i in ns:stat(par['inputs'][str(i)],ns[i],0,'input_gradients')
   else:req(par['inputs'][str(i)] is None,'boolean absent gradients')
  for k,n in paramcounts.items():stat(par['parameters'][k],n,0,'parameter_gradients')
  req(set(row['warm'])==set(paths) and len(row['measured'])==4,'all timings present')
  updates=[]
  for j,path in enumerate(paths):
   u=row['warm'][path];req(events[j+2]==dict(kind='warm',case=name,path=path,**u),'warm journal agreement');update(u,c,w,actor,obs,sup,path);updates.append(u['seconds'])
  ratios=[];timing_rows=[]
  for i,pair in enumerate(row['measured']):
   req(pair['pair']==i and pair['order']==orders[i] and set(pair['paths'])==set(paths),'measured pair metadata')
   for j,path in enumerate(orders[i]):
    u=pair['paths'][path];req(events[4+2*i+j]==dict(kind='measured',case=name,path=path,pair=i,**u),'measured journal agreement');update(u,c,w,actor,obs,sup,path);updates.append(u['seconds'])
   ratio=pair['paths']['original']['seconds']/pair['paths']['shared_columns']['seconds'];req(pair['speed_ratio']==ratio,'speed ratio arithmetic');ratios.append(ratio);timing_rows.append({p:pair['paths'][p]['seconds'] for p in paths})
  median=statistics.median(ratios);threshold=.9 if c['index']<4 else 1.1
  cell={'case':name,'parity_passed':True,'paired_ratios':ratios,'median_speed_ratio':median,'minimum_ratio':threshold,'speed_passed':median>=threshold};req(summary['cells'][c['index']]==cell,'summary cell arithmetic');cells.append(cell)
  comps={k:row[k] for k in ['generation_hash_and_mask_check_seconds','construction_seconds','reset_seconds']};comps['parity_seconds']=par['seconds'];comps['updates_seconds']=sum(updates)
  req(all(finite(v) and v>=0 for v in comps.values()) and finite(row['wall_seconds']) and sum(comps.values())<=row['wall_seconds']+1e-9,'disjoint case time scope')
  for k,v in comps.items():components[k]+=v
  components['case_unallocated_seconds']+=row['wall_seconds']-sum(comps.values());case_total+=row['wall_seconds']
  req(type(row['process_lifetime_peak_rss_bytes']) is int and row['process_lifetime_peak_rss_bytes']>0,'RSS witness');rss_values.append(row['process_lifetime_peak_rss_bytes'])
  rows.append({'case':c,'independently_recomputed_work':w,'actor_shapes':actor,'original_logical_observation_work':obs,'supervised_queries':sup,'paired_seconds':timing_rows,'median_speed_ratio':median,'speed_threshold':threshold,'speed_passed':median>=threshold,'time_scope_seconds':comps,'case_wall_seconds':row['wall_seconds'],'process_lifetime_peak_rss_bytes':row['process_lifetime_peak_rss_bytes']})
 req(rss_values==sorted(rss_values) and rss_values[-1]<=summary['process_lifetime_peak_rss_bytes'],'lifetime RSS monotonic scope')
 req(finite(done['wall_seconds']) and case_total<=done['wall_seconds']<=300,'whole cap and cost containment')
 rss=summary['process_lifetime_peak_rss_bytes'];rsspass=rss<=6*1024**3
 passed=all(c['speed_passed'] and c['parity_passed'] for c in cells) and rsspass
 req(summary['engineering_admission']==done['engineering_admission']==passed and summary['all_parity_passed'] is True and summary['rss_passed']==rsspass and summary['full_training_authorized'] is False,'engineering admission recomputation')
 req(summary['operation_records']==192 and summary['optimizer_updates']==160 and summary['parity_forward_backward_passes']==32,'summary exact work')
 progress={'phase':'run','completed_cases':16}
 for k,v in [('parity_forward',32),('parity_backward',32),('update_forward',160),('update_backward',160),('optimizer_steps',160)]:
  for suffix in ['attempted','returned']:progress[k+'_'+suffix]=v
 req(done['progress']==progress,'completion attempted returned ledger')
 req(len([x for x in operations if x['kind']=='parity'])==32 and len([x for x in operations if x['kind']!='parity'])==160,'journal work counts')
 # Recheck every authenticated local input after analysis. Git tree is immutable by its object name.
 for name,v in list(inputs.items()):req(digest(ROOT/name)==v['sha256'] and (ROOT/name).stat().st_size==v['bytes'],'end input stability')
 result={'status':'completed','study':'dialogue-token-shared-columns-v1','audit_scope':'Independent saved-output hash, public-mask integer counts, scalar timing and gate audit; no imports of experiment, model or numerical framework code.','execution_completed_sha256':COMPLETE,'plan_sha256':PLAN,'pre_run_git_commit':COMMIT,'engineering_admission':passed,'full_training_authorized':False,'counts':{'execution_files':92,'execution_payloads':91,'sources':71,'cases':16,'events':192,'optimizer_updates':160,'parity_forward_backward_passes':32,'speed_cells_passed':sum(x['speed_passed'] for x in cells),'parity_cells_passed':16,'original_minimum_speed_passed':sum(x['speed_passed'] for x in cells[:4]),'original_medium_large_speed_passed':sum(x['speed_passed'] for x in cells[4:12]),'geometry_speed_passed':sum(x['speed_passed'] for x in cells[12:]),'work_fields_per_optimized_call':74,'reference_only_fields':16,'view_only_fields':1,'state_column_logical_view_elements':128,'required_fill_rows_by_layout':[0,3025,3025,3789],'optimized_calls_checked':96,'all_monitor_calls_checked':192,'parent_case_identity_matches':16},'cells':cells,'failed_cells':[c for c in cells if not c['speed_passed']],'rows':rows,'costs':{'whole_seconds':done['wall_seconds'],'case_wall_seconds_sum':case_total,'outside_cases_seconds':done['wall_seconds']-case_total,'disjoint_case_components_seconds':dict(components),'all_update_phase_seconds_by_path':{k:dict(v) for k,v in phase.items()},'process_lifetime_peak_rss_bytes':rss,'maximum_rss_bytes':6*1024**3,'rss_passed':rsspass,'whole_cap_seconds':300,'timing_scope':'Update durations include assembly, forward validation/monitoring, loss, backward/clip, AdamW and scalar readback. Canonical resets and external packing-count comparisons are outside update timings and inside case/whole wall. Whole wall includes final payload hashing, excludes completion-file write and final printing.'},'parity_recorded_witness_maxima':dict(paritymax),'monitor_recorded_max_sum_error_or_overshoot':normmax,'monitor_recorded_mass_above_one_count':overshoots,'monitor_independently_expected_coverage_totals':dict(monitor_totals),'parent_identity_comparisons':parent_pairs,'runtime_metadata':rt,'lineage':lineage,'interpretation':['Engineering admission requires every fixed numerical, speed, work and RSS condition; the recomputed result and all failures are retained.','All sixteen cells, including four minimum and twelve larger workloads, remain in the unchanged conjunctive decision; no subset is selected for admission.','These are sixteen artificial engineering workloads, including one fixed public-mask geometry. There is no task-quality or full-training result, no representative deployment speed estimate and no architecture novelty claim.'],'verification_boundary':['File membership, byte hashes, source identity to the supplied local Git commit, ancestry, public integer geometry/cost counts, event order, phase sums, paired ratios/medians and admission arithmetic were independently recomputed.','Output/input-gradient/parameter-gradient tensors and optimizer states were not retained. Numerical parity, actual tensor contents, runtime execution, clock/RSS measurements and the correspondence between implementation calls and reported counters remain authenticated execution witnesses; they were not independently replayed.','Parent sample/initialization/actor/mask/canonical model-plus-optimizer digests agree in all sixteen cases. Their underlying unsaved tensor values and RNG execution were not regenerated.','Git object byte identity was checked locally. This audit does not independently attest remote publication time or external process execution.','MACs cover the first feature linear layer only. Payload counters separately describe logical float32, bool and int64 sizes, not memory traffic or peak allocations; aliases must not be summed as distinct allocations; the sixteen marked reference-only fields and the logical 128-element strided parameter view are excluded from actual payload byte totals. View counts do not establish allocated bytes or measured autograd savings.'], 'check_counts':dict(checks)}
 def write(p,d):
  with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 write(OUT/'summary.json',result)
 receipt={'auditor_source_sha256':digest(Path(__file__)), 'status':'completed','audit':'independent-saved-output-only','study':result['study'],'execution_completed_sha256':COMPLETE,'plan_sha256':PLAN,'pre_run_git_commit':COMMIT,'engineering_admission':passed,'full_training_authorized':False,'files':{'summary.json':{'sha256':digest(OUT/'summary.json'),'bytes':(OUT/'summary.json').stat().st_size}},'execution_members':{**done['files'],'completed.json':{'sha256':COMPLETE,'bytes':(RUN/'completed.json').stat().st_size}},'source_sha256':source_verified,'authenticated_input_files':inputs,'parent_execution_completed_sha256':PCOMPLETE,'auditor_actions':{'model_calls':0,'optimizer_calls':0,'encoder_calls':0,'corpus_reads':0,'rng_calls':0,'tests_run':0,'new_timings':0,'benchmark_retries':0},'verification_boundary':result['verification_boundary']}
 write(OUT/'receipt.json',receipt)
 print(json.dumps({'status':'completed','summary_sha256':digest(OUT/'summary.json'),'receipt_sha256':digest(OUT/'receipt.json'),'counts':result['counts'],'failed_cells':result['failed_cells'],'costs':result['costs'],'parity':result['parity_recorded_witness_maxima']},indent=2))


if __name__ == '__main__':
 parser=argparse.ArgumentParser(description='Independent saved-output-only shared-columns qualification audit')
 parser.add_argument('--completed-sha256',required=True)
 parser.add_argument('--run',type=Path,default=RUN)
 parser.add_argument('--out',type=Path,default=OUT)
 args=parser.parse_args()
 if len(args.completed_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.completed_sha256):
  parser.error('A lowercase external SHA-256 completion pin is required')
 COMPLETE=args.completed_sha256
 RUN=args.run.resolve(); OUT=args.out.resolve()
 if not RUN.is_relative_to(ROOT) or not OUT.is_relative_to(ROOT):
  parser.error('Audit paths must be within the repository')
 OUT.mkdir(parents=True,exist_ok=False)
 try:
  audit()
 except BaseException as error:
  try:
   failure={'status':'failed','error_type':type(error).__name__,'error':str(error),
    'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'execution_completed_sha256':COMPLETE,'plan_sha256':PLAN,'no_retry':True,
    'scope':'Saved-output audit only; no experiment, model, test, RNG or timing replay'}
   with (OUT/'failed.json').open('x') as stream:
    json.dump(failure,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
  except BaseException as preservation_error:
   error.add_note('Failure receipt preservation also failed: '+repr(preservation_error))
  raise
