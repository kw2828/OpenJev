# Extracted after execution from the exact recorded Python stdin bodies.
# This file was NOT the source artifact at execution time and has NOT been rerun.
# The three sections below preserve the original analysis and two later executed
# interpretation/text amendments. Section bodies are byte-for-byte copies;
# only this header and the section-separator comments were added on extraction.
# Existing diagnostics are exclusive: do not run against the completed outputs.

# ===== EXECUTED STAGE 1: Original saved-only analysis and first output serialization =====
import hashlib,json,math,time
from pathlib import Path
import numpy as np
ROOT=Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
EVAL=ROOT/'output/card-controllers-v1/evaluation-01'
OUT=ROOT/'output/card-controller-posthoc-v1'
COMPLETED='cc0d22c85fe8c9dadb36ce4b6c4fb047ad920c05a929752b1d0df2be94df22ba'
BINDINGS='bbd609e4d9542c6fb1e4bc51dd7144347a96f951c6a091a8a509d916c4d5f49d'
MODES=('kalman','innovation_local','gated_delta')
start=time.monotonic()
def require(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):
    return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def check(base,rel,digest):
    p=(base/rel).resolve()
    require(not Path(rel).is_absolute() and p.is_relative_to(base.resolve()),'Unsafe path')
    require(sha(p)==digest,'Hash mismatch '+rel)
    return p
require(not (OUT/'remaining-errors.json').exists() and not (OUT/'short.md').exists(),'Output exists')
require(sha(EVAL/'completed.json')==COMPLETED,'Completion hash mismatch')
complete=read(EVAL/'completed.json')
require(complete['status']=='complete' and complete['episodes']==3840,'Not complete')
require(len(complete['files'])==7742,'Wrong member count')
require({str(p.relative_to(EVAL)) for p in EVAL.rglob('*') if p.is_file()}==set(complete['files'])|{'completed.json'},'Member set mismatch')
for name,item in complete['files'].items():
    p=check(EVAL,name,item['sha256']); require(p.stat().st_size==item['bytes'],'Size mismatch')
binding_path=check(ROOT,'evidence/card-controllers-v1/source-bindings.json',BINDINGS)
bindings=read(binding_path)
for name,digest in bindings['files'].items():check(ROOT,name,digest)
protocol_path=check(ROOT,'evidence/card-controllers-v1/protocol.json',complete['protocol_sha256'])
protocol=read(protocol_path)
inputs=read(check(ROOT,'evidence/card-controllers-v1/inputs.json',complete['inputs_sha256']))
old=protocol['lineage']['old_bindings']; old_bindings=read(check(ROOT,old['path'],old['sha256']))
for name,digest in old_bindings['files'].items():check(ROOT,name,digest)
# Aggregate exact integer numerators/denominators, not rounded accuracies.
def accuracy(correct,total):return {'correct':correct,'eligible_queries':total,'accuracy':correct/total if total else None}
def describe(values):
    if not values:return {'count':0,'mean':None,'min':None,'max':None}
    return {'count':len(values),'mean':math.fsum(values)/len(values),'min':min(values),'max':max(values)}
metrics=('selected_seen_hidden','selected_seen_hidden_age_gt32','all_seen_hidden','all_seen_hidden_age_gt32','all_seen_hidden_age_le32')
def analyze(arrays,receipt,mode,pair,index):
    obs,actions,ranks,rewards,raw,terminated,truncated=[arrays[k] for k in ('observations','actions','ranks','rewards','raw_probabilities','terminated','truncated')]
    n=len(actions)
    require(obs.shape==(n+1,52) and obs.dtype==np.int64 and np.all(obs[0]==13),'Observation shape/reset')
    require(actions.shape==ranks.shape==rewards.shape==terminated.shape==truncated.shape==(n,),'Transition shape')
    require(actions.dtype==ranks.dtype==np.int64 and rewards.dtype==np.float64 and terminated.dtype==truncated.dtype==np.bool_,'Transition dtype')
    require(raw.shape==(n,52,13) and raw.dtype==np.float64 and np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1)) and np.allclose(raw.sum(2),1,rtol=0,atol=1e-5),'Prediction shape/probabilities')
    require(1<=n<=104 and receipt['native_steps']==n,'Episode length')
    labels=np.full(52,-1,np.int64); last=np.full(52,-1,np.int64)
    selected=np.zeros(52,bool); matched=np.zeros(52,bool); pending=None
    counter={k:[0,0] for k in metrics}
    mismatches=repeat=0; mismatch_pairs=set(); first_all=None; selected_known=selected_visible=0
    for t,a in enumerate(actions):
        a=int(a); current,after=obs[t],obs[t+1]
        require(0<=a<52 and not matched[a] and a!=pending,'Illegal action')
        # Only labels already revealed by frames ending at current observation enter this prediction check.
        mask=(labels>=0)&(current==13)
        ages=t-last
        require(np.all(ages[mask]>=1),'Hidden-label age')
        predictions=np.argmax(raw[t],axis=1)
        correct=predictions==labels
        def tally(key,eligible):
            counter[key][0]+=int(np.count_nonzero(correct&eligible));counter[key][1]+=int(np.count_nonzero(eligible))
        tally('all_seen_hidden',mask)
        tally('all_seen_hidden_age_gt32',mask&(ages>32))
        tally('all_seen_hidden_age_le32',mask&(ages<=32))
        chosen=np.zeros(52,bool);chosen[a]=True
        tally('selected_seen_hidden',mask&chosen)
        tally('selected_seen_hidden_age_gt32',mask&chosen&(ages>32))
        selected_known+=int(labels[a]>=0);selected_visible+=int(current[a]!=13)
        visible=matched.copy();visible[a]=True
        if pending is not None:visible[pending]=True
        require(np.array_equal(after!=13,visible) and np.all((after>=0)&(after<=13)),'Native visible mask')
        known=visible&(labels>=0)
        require(np.array_equal(after[known],labels[known]),'Changing public rank')
        require(ranks[t]==after[a],'Selected rank mismatch')
        if pending is None:
            reward=0.;pending=a
        else:
            same=after[a]==after[pending];reward=2/52 if same else -2/104
            if same:matched[[a,pending]]=True
            else:
                key=tuple(sorted((a,pending)));mismatches+=1;repeat+=int(key in mismatch_pairs);mismatch_pairs.add(key)
            pending=None
        require(rewards[t]==reward,'Native reward mismatch')
        labels[visible]=after[visible];last[visible]=t+1;selected[a]=True
        if first_all is None and np.all(labels>=0):first_all=t+1
        require(bool(terminated[t])==bool(matched.all()) and bool(truncated[t])==(t==103),'Terminal mismatch')
        require(bool(terminated[t] or truncated[t])==(t==n-1),'Terminal prefix mismatch')
    require(np.array_equal(selected,labels>=0),'Visited/public-discovery mismatch')
    success=bool(matched.all());total=math.fsum(rewards)
    require(receipt['success']==success and receipt['return']==total and receipt['matched_pairs']==int(matched.sum())//2,'Receipt outcome mismatch')
    known_unmatched=(labels>=0)&~matched
    available_pairs=sum(int(np.count_nonzero(labels[known_unmatched]==r))//2 for r in range(13))
    result={'mode':mode,'pair':pair,'case_index':index,'seed':receipt['seed'],'success':success,'native_steps':n,'return':total,
            'matched_pairs':int(matched.sum())//2,'unique_visited_positions':int(selected.sum()),'unvisited_positions':int((~selected).sum()),
            'all_positions_discovered':bool(selected.all()),'action_count_at_full_discovery':first_all,
            'actions_after_full_discovery':n-first_all if first_all is not None else None,
            'selected_previously_seen_actions':selected_known,'selected_currently_visible_actions':selected_visible,
            'mismatching_pair_attempts':mismatches,'repeat_mismatching_pair_attempts':repeat,
            'distinct_mismatching_position_pairs':len(mismatch_pairs),'publicly_known_unmatched_positions':int(known_unmatched.sum()),
            'remaining_disjoint_matching_pairs_known_from_public_history':available_pairs}
    result.update({key:accuracy(*counter[key]) for key in metrics})
    return result
rows=[]
for mode in MODES:
    for pair in range(3):
        for index in range(64):
            name=f'{mode}-pair{pair}'; stem=EVAL/'controllers'/name/'C'/'episodes'/f'{index:03d}'
            rec=read(stem.with_suffix('.json'))
            require(rec['status']=='complete' and rec['controller']==name and rec['policy']=='C' and rec['index']==index and rec['seed']==inputs['evaluation'][index]['seed'],'Identity mismatch')
            require(rec['layout_sha256']==complete['layout_sha256_by_case'][str(index)],'Paired deck mismatch')
            require(sha(stem.with_suffix('.npz'))==rec['npz_sha256'],'Episode nested hash')
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:
                arrays={k:z[k] for k in ('observations','actions','ranks','rewards','raw_probabilities','terminated','truncated')}
            rows.append(analyze(arrays,rec,mode,pair,index))
def summarize(items):
    out={'games':len(items),'successes':sum(r['success'] for r in items),'native_steps':sum(r['native_steps'] for r in items),
         'mean_return':math.fsum(r['return'] for r in items)/len(items) if items else None,
         'unique_visited_positions':describe([r['unique_visited_positions'] for r in items]),
         'all_positions_discovered_games':sum(r['all_positions_discovered'] for r in items),
         'undiscovered_at_terminal_games':sum(not r['all_positions_discovered'] for r in items),
         'unvisited_positions':describe([r['unvisited_positions'] for r in items]),
         'matched_pairs':describe([r['matched_pairs'] for r in items]),
         'mismatching_pair_attempts':sum(r['mismatching_pair_attempts'] for r in items),
         'repeat_mismatching_pair_attempts':sum(r['repeat_mismatching_pair_attempts'] for r in items),
         'games_with_repeat_mismatching_pairs':sum(r['repeat_mismatching_pair_attempts']>0 for r in items),
         'publicly_known_unmatched_positions':describe([r['publicly_known_unmatched_positions'] for r in items]),
         'games_with_remaining_publicly_known_matching_pairs':sum(r['remaining_disjoint_matching_pairs_known_from_public_history']>0 for r in items),
         'remaining_disjoint_matching_pairs_known_from_public_history':describe([r['remaining_disjoint_matching_pairs_known_from_public_history'] for r in items]),
         'action_count_at_full_discovery':describe([r['action_count_at_full_discovery'] for r in items if r['all_positions_discovered']]),
         'actions_after_full_discovery':describe([r['actions_after_full_discovery'] for r in items if r['all_positions_discovered']])}
    for key in metrics:
        out[key]=accuracy(sum(r[key]['correct'] for r in items),sum(r[key]['eligible_queries'] for r in items))
        available=[r[key]['accuracy'] for r in items if r[key]['eligible_queries']]
        out[key]['game_macro_accuracy']=math.fsum(available)/len(available) if available else None
        out[key]['games_with_eligible_queries']=len(available)
    return out
summaries={mode:{subset:summarize([r for r in rows if r['mode']==mode and (subset=='all' or r['success']==(subset=='successful'))]) for subset in ('all','successful','failed')} for mode in MODES}
per_fit={f'{mode}-pair{pair}':summarize([r for r in rows if r['mode']==mode and r['pair']==pair]) for mode in MODES for pair in range(3)}
require(len(rows)==576,'Coverage failure')
require(sha(EVAL/'completed.json')==COMPLETED,'Completion changed')
for name,digest in bindings['files'].items():check(ROOT,name,digest)
result={'status':'complete','scope':'exploratory_saved_output_diagnostic_not_a_gate_change',
        'bindings':{'execution_completed_sha256':COMPLETED,'protocol_sha256':complete['protocol_sha256'],'inputs_sha256':complete['inputs_sha256'],
                    'source_bindings_sha256':BINDINGS,'old_source_bindings_sha256':old['sha256'],
                    'authenticated_execution_payloads':7742,'authenticated_execution_payload_bytes':sum(i['bytes'] for i in complete['files'].values()),
                    'authenticated_new_sources':len(bindings['files']),'authenticated_old_sources':len(old_bindings['files'])},
        'coverage':{'policy':'C','modes':list(MODES),'fits_per_mode':3,'episodes_per_fit':64,'episodes_per_mode':192,'episodes':576,'native_public_transitions':sum(r['native_steps'] for r in rows)},
        'definitions':{'time':'Before action t, public observation has index t; returned observation has index t+1.',
            'accuracy':'Argmax over saved raw pre-action model probabilities, first rank wins exact ties; no picker normalization, visible one-hot or unseen-uniform overrides.',
            'labels':'Labels derive exclusively from public frames at indices <=t. No future returned frame or hidden board supplies a prediction target.',
            'eligible_seen_hidden':'Position previously visible in a public frame and currently hidden (token13). Matched cards remain visible and are excluded.',
            'selected_seen_hidden':'The actual selected position satisfies eligible_seen_hidden before that action.',
            'age':'Action-clock difference t minus latest earlier/current public visibility index; age>32 is strict. This is public-visibility age, not necessarily last model-write age.',
            'micro_accuracy':'Sum correct queries divided by all eligible queries, including repeated queries at successive actions. Game macro accuracy averages defined per-game accuracies equally.',
            'visited':'Unique actual selected positions; verified equal to the set revealed anywhere in that game public history.',
            'repeat_mismatching_pairs':'Each mismatching unordered position pair after its first mismatch in the same game counts once; denominator is all mismatching pair attempts.',
            'known_remaining_pairs':'At episode end, for each public rank among known unmatched positions, floor(count/2), summed across ranks. Describes available public information, not a simulated alternative controller.'},
        'by_mode_and_outcome':summaries,'by_fit':per_fit,'per_game':rows,
        'limits':['Post hoc metrics and subsets were chosen after the completed controller result; original architecture and controller gates remain unchanged.',
                  'Success/failure subsets and model trajectories differ. Accuracy differences are descriptive, not causal or a matched-population comparison.',
                  'Repeated queries and three fits share 64 deck draws; query/game counts are not independent statistical replicates.',
                  'Wrong top-rank recall is distinct from action optimality; the picker uses full rank probabilities and pair scores.',
                  'Full discovery does not imply adequate remaining action budget. Known final pairs and repeated mismatches do not establish a counterfactual success.',
                  'No learned forward pass, calibration test, native replay, fitting, seed allocation or RNG call.'],
        'new_model_calls':0,'new_native_calls':0,'new_rng_calls':0,'wall_seconds':time.monotonic()-start}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'remaining-errors.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def pct(a):return f"{100*a['accuracy']:.1f}% ({a['correct']:,}/{a['eligible_queries']:,})" if a['eligible_queries'] else 'not defined (0 queries)'
lines=['# Remaining C-policy errors: exploratory saved-output review','',
       'All 192 completed C-policy games for each of Kalman, local innovation and gated delta are included, covering all three fitted seeds and the same 64 deck draws. The existing gates are unchanged.','',
       '| Model | Successes | Failed games with all 52 positions discovered | Failed games with repeated mismatches | Failed-game selected hidden recall | Failed-game all hidden recall | Failed-game hidden recall, age >32 |',
       '|---|---:|---:|---:|---|---|---|']
for mode in MODES:
    a=summaries[mode]['all']; f=summaries[mode]['failed']
    lines.append(f"| {mode} | {a['successes']}/192 | {f['all_positions_discovered_games']}/{f['games']} | {f['games_with_repeat_mismatching_pairs']}/{f['games']} | {pct(f['selected_seen_hidden'])} | {pct(f['all_seen_hidden'])} | {pct(f['all_seen_hidden_age_gt32'])} |")
lines+=['','Accuracy uses saved raw pre-action probabilities and only ranks revealed by earlier public frames. The age threshold is strictly greater than 32 actions since last public visibility. Each denominator is eligible position-time queries, including repeated queries; per-game macro accuracies and all success/failure subset totals are in the JSON.','',
        '| Model | Successful-game all hidden recall | Successful-game hidden recall, age >32 | Failed-game mean visited positions | Failed-game remaining publicly known pairs, mean |',
        '|---|---|---|---:|---:|']
for mode in MODES:
    s=summaries[mode]['successful'];f=summaries[mode]['failed']
    lines.append(f"| {mode} | {pct(s['all_seen_hidden'])} | {pct(s['all_seen_hidden_age_gt32'])} | {f['unique_visited_positions']['mean']:.2f} | {f['remaining_disjoint_matching_pairs_known_from_public_history']['mean']:.2f} |")
lines+=['','The next mechanism test should separate retrieval quality on identical public histories from controller decisions. These saved trajectories show whether discovery remained incomplete and whether previously exposed ranks were recalled, but cannot establish how correcting one component would change success. Successful and failed games are different selected populations; their accuracy contrast is not a causal effect. Full discovery can also occur too late to finish within 104 actions.','',
        f"Execution completion SHA-256: `{COMPLETED}`.",f"Diagnostic JSON SHA-256: `{sha(OUT/'remaining-errors.json')}`.",'',
        'No model, simulator or RNG calls were made. No main-run, release, source or gate files were changed.','']
with (OUT/'short.md').open('x') as f:f.write('\n'.join(lines))
for mode in MODES:
    print(mode,json.dumps({k:v for k,v in summaries[mode]['failed'].items() if k in ('games','unique_visited_positions','all_positions_discovered_games','games_with_repeat_mismatching_pairs','repeat_mismatching_pair_attempts','mismatching_pair_attempts','games_with_remaining_publicly_known_matching_pairs','selected_seen_hidden','all_seen_hidden','all_seen_hidden_age_gt32','action_count_at_full_discovery','actions_after_full_discovery')},sort_keys=True))
print('wall_seconds',result['wall_seconds'])
for p in (OUT/'remaining-errors.json',OUT/'short.md'):print(p,sha(p),p.stat().st_size)

# ===== EXECUTED STAGE 2: Added interpretation and explanatory prose; no metric recomputation =====
import json,hashlib
from pathlib import Path
p=Path('output/card-controller-posthoc-v1/remaining-errors.json')
j=json.loads(p.read_text())
j['interpretation']={
 'observed':'Every failed game in these three families left at least one card undiscovered. Failed-game mean unique discovery is 46.59 to 46.90 of 52 positions, while raw all-seen-hidden rank accuracy is 99.37% to 99.49%.',
 'remaining_symptoms':'Repeated mismatching position pairs occur in 81.65% to 84.68% of failed games. Recall errors remain, especially on older eligible queries, and final publicly known matching pairs remain in 21 failed games per family.',
 'next_test_implication':'Prioritize a bounded controller exploration and repeated-mismatch investigation with unchanged weights before interpreting this failure pattern as a need for a new memory architecture. Any new evaluation needs a separately fixed rule and fresh cases; these completed outcomes cannot serve as an untouched confirmation.',
 'not_established':'Incomplete discovery is a necessary observed barrier to full completion, not a causal decomposition. Better recall or different probability magnitudes could alter later discovery. The saved analysis does not simulate a corrected controller or establish which intervention would help most.'}
p.write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
md=p.with_name('short.md'); s=md.read_text()
old='The next mechanism test should separate retrieval quality on identical public histories from controller decisions. These saved trajectories show whether discovery remained incomplete and whether previously exposed ranks were recalled, but cannot establish how correcting one component would change success.'
new=('Every failed game left at least one position undiscovered. Failed-game raw hidden-card recall nevertheless reaches 99.37% to 99.49%, while repeated mismatches occur in 81.65% to 84.68% of failures. Final public histories still contain known matching pairs in 21 failed games per family. These observations favor investigating exploration and repeated decision waste with unchanged weights before attributing the remaining failures to memory architecture. Older-card errors remain and could influence later exploration.\n\n'
     'A future comparison needs a separately fixed controller rule and fresh cases. This analysis cannot establish how correcting recall or changing decisions would affect success.')
assert old in s;s=s.replace(old,new)
import re
s=re.sub(r'Diagnostic JSON SHA-256: `[^`]+`\.',f'Diagnostic JSON SHA-256: `{sha(p)}`.',s)
md.write_text(s)
for mode,sets in j['by_mode_and_outcome'].items():
 for subset in ('all','successful','failed'):
  a=sets[subset];print(mode,subset,a['games'],a['all_seen_hidden'],a['all_seen_hidden_age_gt32'])
for f in(p,md):print(f,sha(f))

# ===== EXECUTED STAGE 3: Corrected displayed repeat-rate range and clarified causal/confidence limits =====
import json,hashlib,re
from pathlib import Path
p=Path('output/card-controller-posthoc-v1/remaining-errors.json');j=json.loads(p.read_text())
j['interpretation']['remaining_symptoms']=j['interpretation']['remaining_symptoms'].replace('81.65% to 84.68%','80.36% to 84.68%')
j['interpretation']['not_established'] += ' A small number of recall errors can itself cause repeated choices and incomplete discovery; repeated-pair confidence was not analyzed.'
p.write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
md=p.with_name('short.md');s=md.read_text().replace('81.65% to 84.68%','80.36% to 84.68%')
s=s.replace('Older-card errors remain and could influence later exploration.','A small number of recall errors could itself cause repeated choices and incomplete discovery. Confidence on repeated mismatching pairs was not analyzed.')
s=re.sub(r'Diagnostic JSON SHA-256: `[^`]+`\.',f'Diagnostic JSON SHA-256: `{sha(p)}`.',s);md.write_text(s)
for p in(p,md):print(p,sha(p))
