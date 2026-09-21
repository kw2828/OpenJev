"""One frozen saved-only calibration-preparation audit; no tokenizer or model calls.

Authentication and native timing reuse pinned common helpers. Selection, actor
maps, source/evaluator joins, original lexical values and workload arithmetic
are independently reconstructed. Number-normalized match flags 0..3 inherit
the frozen producer; their binary/support/carry and invariant columns are checked.
No predictions, checkpoint tensors, calibration scores or TEST are decoded.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import signal
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
COMMON_PIN = '452016fcdb611199fa4fafcbb54eeb4ab92cdc6f54b94a17d2802eb2223b90dd'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
PLAN = ROOT / 'output/dialogue-calibration-control-v1/plan.json'
PLAN_PIN = 'e706e40cd4a0d95fb1536ce105074c071b57a6bc4652ec57b9243672542a413c'
PREP = ROOT / 'runs/dialogue-calibration-control-v1/preparation-01'
PREP_PIN = '896099f38f7316f23e549d1271d81f90a2884028e6c0281dcf3c035b6ee286e9'
TERMINAL = ROOT / 'output/dialogue-calibration-control-v1/preparation-process-01.terminal.json'
TERMINAL_PIN = '9a3f54de5ee09259fdfafa9e569556fc51ff78c3bf7ea8b29842e9944aa05bd3'
LIMITS = {'wall_seconds': 300, 'rss_bytes': 8*1024**3, 'output_bytes': 32*1024**2}
NONE, DC = 'reserved:NOT_MENTIONED', 'reserved:DONTCARE'
STRATA = ('unmentioned_retention', 'assigned_retention', 'changed')
ACTOR = {'split','dialogue_id','query_ids','user_turn_indices','tokens','original_feature_ids',
         'turn_text_ids','query_text_ids','candidate_text_ids','candidate_ids','source_split',
         'analysis_role','lexical_offset','lexical_shape'}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def need(x, reason):
    if not x:
        raise ValueError(reason)


def lines(p, check):
    values = []
    with Path(p).open() as f:
        for i, line in enumerate(f):
            if i % 256 == 0:
                check()
            need(bool(line.strip()), 'Nonempty JSONL records')
            values.append(json.loads(line))
    return values


def index(rows, key):
    answer = {r[key]:r for r in rows}
    need(len(answer) == len(rows), 'Unique ' + key)
    return answer


def public_pairs(record):
    need(set(record) == {'dialogue_id','turns','user_turns'}, 'Public dialogue fields')
    systems, users, chronology = [], [], []
    for i, t in enumerate(record['turns']):
        need(set(t) == {'speaker','utterance'} and t['speaker'] in ('USER','SYSTEM')
             and type(t['utterance']) is str, 'Public role/text')
        if t['speaker'] == 'USER':
            si = i-1 if i and record['turns'][i-1]['speaker'] == 'SYSTEM' else None
            systems.append('' if si is None else record['turns'][si]['utterance'])
            users.append(t['utterance'])
            chronology.append({'turn_index': i, 'previous_system_turn_index': si})
    need(chronology == record['user_turns'] and users, 'All public USER positions')
    return systems, users, [x['turn_index'] for x in chronology]


def identity(record):
    normalized = [[t['speaker'], ' '.join(unicodedata.normalize('NFKC', t['utterance']).casefold().split())]
                  for t in record['turns']]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def rank(did):
    return hashlib.sha256(('openjev-calibration-v1:' + did).encode()).hexdigest(), did


def work(actor):
    tokens = actor['tokens']
    lengths = [min(254, len(ids)-i)+2 for ids in tokens for i in range(0, len(ids), 254)]
    groups = [lengths[i:i+32] for i in range(0,len(lengths),32)]
    t, q = len(actor['turn_text_ids']), len(actor['query_text_ids'])
    c = max(map(len, actor['candidate_ids']))
    padded = sum(len(g)*max(g) for g in groups)
    lexical = t*q*c*10
    return {'unique_texts':len(tokens),'input_texts':len(tokens),'content_tokens':sum(map(len,tokens)),
            'chunks':len(lengths),'encoder_sequences':len(lengths),'special_token_positions':2*len(lengths),
            'valid_token_positions':sum(lengths),'padded_token_positions':padded,
            'padding_token_positions':padded-sum(lengths),
            'padded_attention_positions':sum(len(g)*max(g)**2 for g in groups),'encoder_calls':len(groups),
            'chunk_tokens':254,'chunk_batch_size':32,'overlength_texts_chunked':sum(len(x)>254 for x in tokens),
            'truncated_tokens':0,'max_chunk_tokens_with_special':max(lengths),'public_user_turns':t,
            'queries':q,'schema_text_occurrences':q,'candidate_text_occurrences':sum(map(len,actor['candidate_ids'])),
            'max_candidates':c,'real_question_updates':t*q,
            'real_candidate_updates':t*sum(map(len,actor['candidate_ids'])),
            'padded_candidate_positions':t*q*c,'lexical_scalars':lexical,'lexical_bytes':4*lexical,
            'max_content_tokens_per_text':max(map(len,tokens))}


def reduce_work(profiles):
    settings = {'chunk_tokens','chunk_batch_size'}
    keys = set(profiles[0]['work'])-settings
    return {'dialogues':len(profiles),
            'totals':{k:sum(p['work'][k] for p in profiles) for k in keys if not k.startswith('max_')},
            'maxima':{k:max(p['work'][k] for p in profiles) for k in keys}}


def original_lexical(systems, users, ids, values, np):
    result = np.zeros((len(users),len(ids),10),dtype=np.float32)
    result[:,ids.index(NONE),6] = 1
    result[:,ids.index(DC),7] = 1
    values = {i:v.strip().casefold() for i,v in enumerate(values) if v is not None}
    pats = {i:re.compile(r'(?<!\w)'+re.escape(v)+r'(?!\w)') for i,v in values.items()}
    current = ids.index(NONE)
    for ti,(system,user) in enumerate(zip(systems,users,strict=True)):
        winners = []
        for col,text in enumerate((user,system)):
            hits = [i for i,p in pats.items() if p.search(text.casefold())]
            result[ti,hits,col] = 1
            best = [i for i in hits if len(values[i]) == max(len(values[j]) for j in hits)] if hits else []
            winner = best[0] if len(best) == 1 else None
            winners.append(winner)
            if winner is not None:
                result[ti,winner,col+2] = 1
        result[ti,current,5] = 1
        if winners[0] is not None:
            current = winners[0]
        result[ti,current,4] = 1
        positive = bool(re.search(r'(?<!\w)(?:yes|yeah|yep|true)(?!\w)',user.casefold()))
        negative = bool(re.search(r'(?<!\w)(?:no|nope|false)(?!\w)',user.casefold()))
        for i,v in values.items():
            result[ti,i,8] = v == 'true' and positive
            result[ti,i,9] = v == 'false' and negative
    return result


def inspect(ctx, done, check, np):
    source, original = ROOT/'runs/sgd-state-v1/data', ctx['original_prepared']
    train = index(lines(source/'train-dialogues.jsonl',check),'dialogue_id')
    dev = index(lines(source/'dev-dialogues.jsonl',check),'dialogue_id')
    labels = lines(source/'train-labels.jsonl',check)
    catalogs = read(source/'catalog.json')
    catalog = catalogs['train']
    cat = index(catalog,'query_id')
    old_actors = lines(original/'actors-dev.jsonl',check)
    old_index = index(old_actors,'dialogue_id')
    fitted = read(original/'orders.json')['dialogue_ids']
    need(len(fitted) == len(set(fitted)) == 2017 and set(fitted)<=set(train), 'Original fitted membership')
    need(len(old_index) == 2363 and set(old_index)<=set(dev), 'Original DEV membership')
    turns = {}
    identities = {}
    for did,p in train.items():
        check()
        turns[did] = set(public_pairs(p)[2])
        identities[did] = identity(p)
    dev_groups = set()
    for did in old_index:
        check()
        public_pairs(dev[did])
        dev_groups.add(identity(dev[did]))
    fitted_groups = {identities[d] for d in fitted}
    eligible, endpoints, by_did = set(), set(), defaultdict(list)
    for r in labels:
        did, qi, ti = r['dialogue_id'],r['query_id'],r['turn_index']
        need(did in train and qi in cat and type(ti) is int and ti in turns[did], 'Categorical endpoint identity')
        need((did,qi,ti) not in endpoints, 'Unique source endpoint')
        endpoints.add((did,qi,ti)); eligible.add(did); by_did[did].append(r)
    fc = {d for d in eligible if identities[d] in fitted_groups}
    dc = {d for d in eligible if identities[d] in dev_groups}
    survivors = eligible-fc-dc
    grouped = defaultdict(list)
    for did in survivors:
        grouped[identities[did]].append(did)
    reps = sorted((min(g,key=rank) for g in grouped.values()),key=rank)
    need(len(reps)>=512, 'Enough calibration groups')
    chosen = reps[:512]
    counts = {'train_public_dialogues':len(train),'fitted_ids':len(fitted),'evaluated_dev_dialogues':len(old_index),
              'eligible_categorical_dialogues':len(eligible),'ineligible_without_categorical_endpoints':len(train)-len(eligible),
              'excluded_fitted_group_eligible_dialogues':len(fc),'excluded_dev_group_eligible_dialogues':len(dc),
              'excluded_either_group_eligible_dialogues':len(fc|dc),'excluded_both_groups_eligible_dialogues':len(fc&dc),
              'excluded_fitted_eligible_groups':len({identities[d] for d in fc}),
              'excluded_dev_eligible_groups':len({identities[d] for d in dc}),
              'surviving_eligible_dialogues':len(survivors),'surviving_unique_groups':len(grouped),
              'duplicate_members_not_represented':len(survivors)-len(grouped),'selected_dialogues':512,
              'selected_unique_groups':512,'unselected_unique_groups':len(grouped)-512}
    selection = read(PREP/'selection.json')
    need(selection['selected_ids']==chosen and selection['counts']==counts==done['selection_counts'], 'Exact salted cohort/counts')
    need(selection['sample_size']==512 and selection['salt']=='openjev-calibration-v1:'
         and selection['source_split']=='train' and selection['analysis_role']=='calibration'
         and selection['selection_fields']==['dialogue_id','query_id','turn_index'], 'Selection scope')
    need(selection['selected_groups']==[{'dialogue_id':d,'text_sha256':identities[d],'salted_id_sha256':rank(d)[0]}
                                       for d in chosen], 'Every chosen hash and full-text group')
    packet = read(PREP/'packet.json')
    actors = lines(PREP/'actors.jsonl',check)
    targets = lines(PREP/'targets.jsonl',check)
    rows = lines(PREP/'evaluation-rows.jsonl',check)
    need(len(actors)==len(targets)==512 and [a['dialogue_id'] for a in actors]==chosen, 'All actor order/membership')
    text_list, text_map, queries = [], {}, []
    def add_text(text):
        if text not in text_map:
            text_map[text]=len(text_list); text_list.append(text)
        return text_map[text]
    for q in catalog:
        prefix = 'Service: '+q['service_description']+'\nSlot: '+q['slot_description']
        need(q['query_text']==prefix and json.loads(q['query_id'])==[q['service'],q['slot']], 'Canonical query semantics')
        ids, values = [c['id'] for c in q['candidates']], [c['value'] for c in q['candidates']]
        need(3<=len(ids)<=12 and len(set(ids))==len(ids) and ids.count(NONE)==ids.count(DC)==1, 'Complete candidate support')
        for c in q['candidates']:
            cid,v = c['id'],c['value']
            value = 'NOT_MENTIONED (no constraint stated)' if cid==NONE else 'DONTCARE (no preference)' if cid==DC else v
            need((v is None if cid in (NONE,DC) else type(v) is str and cid=='value:'+v)
                 and c['text']==prefix+'\nValue: '+value, 'Canonical candidate text and ID')
        queries.append({'id':q['query_id'],'split':'train','service':q['service'],'slot':q['slot'],
                        'text':add_text(prefix),'candidates':[add_text(c['text']) for c in q['candidates']],
                        'candidate_ids':ids,'candidate_values':values})
    qi_by_id = {q['id']:i for i,q in enumerate(queries)}
    need(packet['queries']==queries, 'Full TRAIN catalog order and text indices')
    lexical = {v:np.load(PREP/f'lexical-{v}.npy',allow_pickle=False,mmap_mode='r') for v in ('original','numbers')}
    need(all(a.dtype==np.float32 and a.ndim==1 for a in lexical.values()), 'Lexical flat f32 arrays')
    expected_rows, layouts, cohort, profiles, token_consistency = [], [], [], [], {}
    offset = user_count = 0
    vocab = read(ctx['parent']['model_files']['config.json']['path'])['vocab_size']
    for a,tr,did in zip(actors,targets,chosen,strict=True):
        check()
        need(set(a)==ACTOR and a['split']==a['source_split']=='train' and a['analysis_role']=='calibration', 'Exact public actor whitelist')
        need(set(tr)=={'split','source_split','analysis_role','dialogue_id','rows','literal_registers'}
             and tr['dialogue_id']==did and tr['split']==tr['source_split']=='train'
             and tr['analysis_role']=='calibration', 'Separate evaluator identity')
        systems,users,ut = public_pairs(train[did]); user_count += len(users)
        global_turns = [add_text('System: '+s+'\nUser: '+u) for s,u in zip(systems,users,strict=True)]
        qids = sorted({qi_by_id[r['query_id']] for r in by_did[did]})
        ids = [queries[qi]['candidate_ids'] for qi in qids]
        shape = [len(users),len(qids),max(map(len,ids)),10]
        layout = {'id':did,'query_ids':qids,'offset':offset,'shape':shape}
        layouts.append(layout); cohort.append({'id':did,'turns':global_turns,'query_ids':qids})
        need(a['query_ids']==qids and a['candidate_ids']==ids and a['user_turn_indices']==ut
             and a['lexical_offset']==offset and a['lexical_shape']==shape, 'Public chronology/schema/layout')
        feature_order, seen_feature = [], {}
        def local(fi):
            if fi not in seen_feature:
                seen_feature[fi]=len(feature_order); feature_order.append(fi)
            return seen_feature[fi]
        turn_local = [local(fi) for fi in global_turns]
        query_local, candidate_local = [], []
        for qi in qids:
            query_local.append(local(queries[qi]['text']))
            candidate_local.append([local(fi) for fi in queries[qi]['candidates']])
        need(a['original_feature_ids']==feature_order and a['turn_text_ids']==turn_local
             and a['query_text_ids']==query_local and a['candidate_text_ids']==candidate_local, 'All local/global text maps')
        need(len(a['tokens'])==len(feature_order), 'Every mapped unique text tokenized')
        for fi,tokens in zip(feature_order,a['tokens'],strict=True):
            need(type(tokens) is list and tokens and all(type(t) is int and 0<=t<vocab for t in tokens), 'Token integer/range support')
            digest = hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
            need(fi not in token_consistency or token_consistency[fi]==digest, 'Identical public text tokenization')
            token_consistency[fi]=digest
        size = math.prod(shape)
        slices = {v:ar[offset:offset+size].reshape(shape) for v,ar in lexical.items()}
        for ar in slices.values():
            need(np.isfinite(ar).all() and np.isin(ar,(0,1)).all(), 'Binary finite lexical observations')
        need(np.array_equal(slices['original'][...,6:],slices['numbers'][...,6:]), 'Reserved/Boolean columns unchanged')
        literal = {v:[] for v in lexical}
        for j,qi in enumerate(qids):
            cids, values = ids[j],queries[qi]['candidate_values']; n=len(cids)
            original_flags = original_lexical(systems,users,cids,values,np)
            need(np.array_equal(original_flags,slices['original'][:,j,:n]), 'Independent original lexical observations')
            for v,ar in slices.items():
                need(not ar[:,j,n:].any() and np.all(ar[:,j,:n,4].sum(-1)==1)
                     and np.all(ar[:,j,:n,5].sum(-1)==1), 'Padding and single literal registers')
                register = ar[:,j,:n,4].argmax(-1).tolist(); literal[v].append(register)
                need(ar[0,j,cids.index(NONE),5]==1 and np.array_equal(ar[1:,j,:n,5],ar[:-1,j,:n,4]), 'Causal register carry')
                for t in range(len(users)):
                    winners = np.flatnonzero(ar[t,j,:n,2]).tolist()
                    need(len(winners)<=1, 'At most one unique longest match')
                    previous = cids.index(NONE) if t==0 else register[t-1]
                    need(register[t]==(winners[0] if winners else previous), 'Literal write only from unique USER match')
        need(tr['literal_registers']==literal, 'Evaluator literal reference from actor flags')
        expected_target = []
        for source_i,r in enumerate(by_did[did]):
            qi=qi_by_id[r['query_id']]; j=qids.index(qi); t=ut.index(r['turn_index'])
            label=r['label_index']; q=queries[qi]
            need(type(label) is int and 0<=label<len(q['candidate_ids']) and q['candidate_ids'][label]==r['label_id']
                 and r['service']==q['service'] and r['slot']==q['slot'] and r['unseen_service'] is False
                 and type(r['is_dontcare']) is bool and r['is_dontcare']==(r['label_id']==DC), 'Original annotation identity only')
            stratum = r['bin'] if r['bin'] in STRATA[:2] else 'changed'
            x = {'split':'train','dialogue_id':did,'source_row_index':source_i,'time':t,'turn_index':r['turn_index'],
                 'query_index':qi,'query_position':j,'query_id':r['query_id'],'service':r['service'],'slot':r['slot'],
                 'label_index':label,'label_id':r['label_id'],'bin':r['bin'],'stratum':stratum,
                 'stratum_index':STRATA.index(stratum),'unseen':False,'dontcare':r['is_dontcare']}
            expected_target.append(x)
            expected_rows.append({**x,'row_index':len(expected_rows),'candidate_ids':q['candidate_ids'],
                                  'candidate_values':q['candidate_values'],'source_split':'train','analysis_role':'calibration'})
        need(tr['rows']==expected_target, 'Exact separate targets/source order')
        previous = {}
        for r in sorted(expected_target,key=lambda r:(r['time'],r['query_index'])):
            before=previous.get(r['query_index'],NONE); current=r['label_id']
            bin_name = ('unmentioned_retention' if current==NONE else 'assigned_retention') if current==before else \
                       'first_assignment' if before==NONE else 'clear' if current==NONE else 'revision'
            need(r['bin']==bin_name, 'Evaluator transition metadata')
            previous[r['query_index']]=current
        profiles.append({'split':'train','dialogue_id':did,'work':work(a)})
        offset += size
    need(rows==expected_rows and packet=={'queries':queries,'cohort':cohort,'scope':'Public indices only; targets separate',
                                         'texts':text_list,'layouts':layouts}, 'Complete actor packet and evaluator rows')
    need(all(len(ar)==offset for ar in lexical.values()) and len(token_consistency)==len(text_list), 'Exact lexical/text coverage')
    final_counts={'dialogues':512,'scored_endpoints':len(rows),'public_user_turns':user_count,
                  'lexical_positions':offset,'unique_texts':len(text_list)}
    need(done['counts']==final_counts and done['unique_tokenized_texts']==len(text_list), 'Preparation terminal counts')
    geometry=read(PREP/'workloads.json'); reduced=reduce_work(profiles)
    need(geometry['profiles']==profiles and geometry['all']==reduced and geometry['splits']=={'train':reduced}
         and geometry['chunk_tokens']==254 and geometry['chunk_batch_size']==32, 'Every profile plus sums/maxima')
    old_profiles={p['dialogue_id']:p for p in read(original/'workloads.json')['profiles'] if p['split']=='dev'}
    need(set(old_profiles)==set(old_index), 'All DEV profiles')
    for a in old_actors:
        check()
        need(a['split']=='dev' and work(a)==old_profiles[a['dialogue_id']]['work']
             and a['user_turn_indices']==public_pairs(dev[a['dialogue_id']])[2], 'Independent complete DEV geometry/chronology')
    first=old_actors[0]['dialogue_id']
    largest=min(set(old_index)-{first},key=lambda d:(-old_profiles[d]['work']['padded_attention_positions'],d))
    replay=read(PREP/'replay-cases.json')
    need([c['dialogue_id'] for c in replay['cases']]==[first,largest]==done['replay_dialogue_ids'], 'Two fixed DEV identities')
    old_rows=lines(ctx['prior'].run/'evaluation-rows.jsonl',check)
    need(len(old_rows)==62329 and replay['evaluation_rows_sha256']==sha(ctx['prior'].run/'evaluation-rows.jsonl')
         and Path(replay['evaluation_rows_source'])==ctx['prior'].run/'evaluation-rows.jsonl', 'Original DEV evaluator hash')
    endpoint_fields=('row_index','source_row_index','time','turn_index','query_position','query_index','query_id',
                     'service','slot','candidate_ids','candidate_values')
    dev_catalog=index(catalogs['dev'],'query_id')
    need(all(r['row_index']==i and r['split']=='dev' and r['dialogue_id'] in old_index
             and type(r['unseen']) is bool for i,r in enumerate(old_rows)), 'Complete canonical DEV rows')
    for case,reason in zip(replay['cases'],('first_original_actor','largest_remaining_attention'),strict=True):
        did=case['dialogue_id']; actor=old_index[did]
        refs=[{k:r[k] for k in endpoint_fields} for r in old_rows if r['dialogue_id']==did]
        need(case=={'dialogue_id':did,'reason':reason,'work':old_profiles[did]['work'],
                    'row_indices':[r['row_index'] for r in refs],'endpoints':refs}, 'Exact public replay endpoints')
        for r in refs:
            j,t=r['query_position'],r['time']
            schema=dev_catalog[r['query_id']]
            need(r['candidate_ids']==actor['candidate_ids'][j] and r['query_index']==actor['query_ids'][j]
                 and r['turn_index']==actor['user_turn_indices'][t]
                 and r['candidate_ids']==[c['id'] for c in schema['candidates']]
                 and r['candidate_values']==[c['value'] for c in schema['candidates']]
                 and (r['service'],r['slot'])==(schema['service'],schema['slot']), 'Replay candidate/time/public schema joins')
    services={r['service'] for r in rows}; query_ids={r['query_id'] for r in rows}
    unseen={r['service'] for r in old_rows if r['unseen']}; seen={r['service'] for r in old_rows if not r['unseen']}
    need(not seen&unseen, 'Stable old DEV exposure')
    unseen_queries={r['query_id'] for r in old_rows if r['unseen']}; overlap=services&unseen
    exposure={'calibration_train_services':sorted(services),'calibration_train_queries':sorted(query_ids),
              'original_dev_seen_services':sorted(seen),'original_dev_unseen_services':sorted(unseen),
              'original_dev_unseen_queries':sorted(unseen_queries),
              'calibration_overlap_with_dev_seen_services':sorted(services&seen),
              'calibration_overlap_with_dev_unseen_services':sorted(overlap),
              'calibration_overlap_with_dev_unseen_queries':sorted(query_ids&unseen_queries),
              'counts':{'calibration_train_services':len(services),'calibration_train_queries':len(query_ids),
                        'original_dev_seen_services':len(seen),'original_dev_unseen_services':len(unseen),
                        'overlap_with_dev_seen_services':len(services&seen),'overlap_with_dev_unseen_services':len(overlap),
                        'overlap_with_dev_unseen_queries':len(query_ids&unseen_queries),
                        'calibration_endpoints_in_dev_unseen_services':sum(r['service'] in overlap for r in rows),
                        'calibration_dialogues_in_dev_unseen_services':len({r['dialogue_id'] for r in rows if r['service'] in overlap}),
                        'dev_unseen_endpoints_in_calibration_services':sum(r['unseen'] and r['service'] in services for r in old_rows)},
              'scope':'Original DEV seen/unseen labels retained; calibration selection never uses this exposure summary'}
    need(read(PREP/'service-exposure.json')==exposure==done['service_exposure'], 'Full service exposure accounting')
    return {'selection_counts':counts,'counts':final_counts,'work':reduced,'service_exposure_counts':exposure['counts'],
            'replay_dialogue_ids':[first,largest],'replay_endpoint_counts':[len(c['endpoints']) for c in replay['cases']],
            'old_dev_profiles_independently_recomputed':len(old_profiles),
            'all_actor_fields_and_maps_checked':True,'original_lexical_independently_recomputed':True,
            'number_lexical_scope':'binary/support/padding/carry and columns6..9; matching flags0..3 inherit frozen source',
            'token_scope':'all IDs integral/in-vocabulary, complete mapped coverage and cross-text consistency; no retokenization',
            'no_model_prediction_arrays_decoded':True,'no_calibration_fit_or_quality_scoring':True}


def main():
    source_pin=sha(__file__)
    need(not (OUT/'started.json').exists(), 'Exclusive audit invocation')
    clock=deadline=None; handler=None; elapsed=None; progress={'stage':'initialization'}
    try:
        need(sha(ROOT/'scripts/dialogue_calibration_common.py')==COMMON_PIN, 'Common source pin')
        need(sha(ROOT/'src/openjev/research/suspend_clock.py')==CLOCK_PIN, 'Native clock source pin')
        import dialogue_calibration_common as common
        import numpy as np
        clock=common.SuspendClock(); deadline=clock.deadline_after(300)
        def check():
            nonlocal elapsed
            elapsed=clock.now_ns()-deadline.started_ns
            need(elapsed<300*10**9, 'Audit native elapsed cap')
            need(common.peak_rss()<=LIMITS['rss_bytes'], 'Audit RSS cap')
            need(sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=LIMITS['output_bytes'], 'Audit output cap')
        def timeout(*_):
            raise TimeoutError('Audit wall alarm')
        handler=signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,300)
        write(OUT/'started.json',{'status':'started','source_sha256':source_pin,'plan_sha256':PLAN_PIN,
              'preparation_completed_sha256':PREP_PIN,'terminal_sha256':TERMINAL_PIN,'limits':LIMITS,
              'clock_backend':clock.backend,'started_ns':deadline.started_ns,'deadline_ns':deadline.expires_ns,
              'rows_decoded_before_source_freeze':False,'experimental_model_calls':0,'tokenizer_calls':0})
        budget=SimpleNamespace(check=check,progress=progress)
        args=SimpleNamespace(plan=PLAN,plan_sha256=PLAN_PIN,out=PREP,command='prepare')
        ctx=common.authenticate(args,budget)
        done=common.authenticate_output(PREP,PREP_PIN,ctx,'prepare')
        terminal=common.authenticate_terminal(TERMINAL,TERMINAL_PIN,done)
        need(len(ctx['plan']['sources'])==13 and len(ctx['science']['source_sha256'])==64,'13 new and64 inherited sources')
        need(done['model_calls']==done['encoder_calls']==done['neural_calls']==0 and done['tokenizer_only'] is True
             and done['model_weights_loaded'] is False and done['official_test_opened'] is False, 'Model-free preparation witness')
        progress['stage']='independent_rows'
        summary=inspect(ctx,done,check,np)
        progress['stage']='final_binding'
        common.bind(ctx['plan']['sources'],check); common.bind(ctx['science']['source_sha256'],check)
        common.authenticate_output(PREP,PREP_PIN,ctx,'prepare')
        need(sha(__file__)==source_pin and sha(PLAN)==PLAN_PIN and sha(TERMINAL)==TERMINAL_PIN, 'Stable audit/input bytes')
        check()
        write(OUT/'summary.json',summary)
        check()
        receipt={'status':'completed','agreement':True,'source_sha256':source_pin,'plan_sha256':PLAN_PIN,
                 'preparation_completed_sha256':PREP_PIN,'terminal_sha256':TERMINAL_PIN,
                 'parent_actual_returncode':terminal['returncode'],'parent_group_absent':terminal['group_absent'],
                 'parent_wall_seconds':terminal['wall_seconds'],'source_count_new':13,'source_count_inherited':64,
                 'preparation_payloads':done['files'],'experimental_model_calls':0,'tokenizer_calls':0,
                 'prediction_arrays_decoded':0,'checkpoint_tensors_loaded':0,'official_test_opened':False,
                 'authentication_scope':'Reused pinned common authentication of source/runtime/whole old byte lineage and new exact payload closure',
                 'independent_scope':'Full salted selection, text-group exclusions, public actor maps, target/evaluator order, original lexical, token geometry, replay choice, service exposure',
                 'limits':LIMITS,'clock_backend':clock.backend,'started_ns':deadline.started_ns,'deadline_ns':deadline.expires_ns,
                 'elapsed_ns':elapsed,'wall_seconds':elapsed/1e9,'peak_rss_bytes':common.peak_rss(),
                 'timing_scope':'Native start through pre-receipt check; receipt publication checked again before return',
                 'files':{n:{'sha256':sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in ('audit.py','started.json','summary.json')}}
        write(OUT/'receipt.json',receipt); check()
        print(json.dumps({'status':'completed','agreement':True,'receipt_sha256':sha(OUT/'receipt.json'),
                          'summary_sha256':sha(OUT/'summary.json'),'wall_seconds':elapsed/1e9,'counts':summary['counts']}))
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        if (OUT/'receipt.json').exists():
            (OUT/'receipt.json').rename(OUT/'invalid-receipt.json')
        try:
            write(OUT/'failed.json',{'status':'failed','source_sha256':source_pin,'error_type':type(error).__name__,
                  'error':str(error),'progress':progress,'last_successful_elapsed_ns':elapsed,
                  'experimental_model_calls':0,'tokenizer_calls':0})
        except BaseException as cleanup:
            error.add_note('Failure receipt error: '+repr(cleanup))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if handler is not None:
            signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    main()
