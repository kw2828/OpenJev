"""Independent saved-metadata-only reconstruction; no producer/model imports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
RUN = ROOT / 'runs/dialogue-conditional-v1/preparation-01'
FREEZE = ROOT / 'output/dialogue-conditional-v1/preparation-protocol-01'
COMPLETION = '960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3'
PLAN = 'fa68f62c708abf545785aa61eab9a54494c9d3bb9979bbf34891a6a473ba8f47'
COMMIT = '37361112e90e4cd3143254a074659e930240c133'
VERSION = 'dialogue-conditional-preparation-v1'
NONE, DC = 'reserved:NOT_MENTIONED', 'reserved:DONTCARE'
REASONS = ['admitted', 'first_public_turn', 'missing_adjacent_scored_predecessor', 'previous_candidate_absent']
BINS = ['unmentioned_retention', 'assigned_retention', 'first_assignment', 'revision', 'clear']
INTEGER_NAMES = ['context-indices.npy', 'offsets.npy', 'chunk-offsets.npy', 'chunk-lengths.npy', 'text-token-counts.npy']
FLOAT_NAMES = {'features.npy', 'lexical.npy', 'tokens.npy', 'priors.npy'}
VERIFIED = {}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def strict_pairs(pairs):
    d = {}
    for k, v in pairs:
        check(k not in d, 'Duplicate JSON key')
        d[k] = v
    return d


def decode(text):
    def bad(x):
        raise ValueError('Invalid JSON number ' + x)
    return json.loads(text, object_pairs_hook=strict_pairs, parse_constant=bad)


def read(path):
    return decode(Path(path).read_text())


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def same(a, b, label):
    check(canonical(a) == canonical(b), label)


def record(path, pin, size=None):
    path = Path(path).resolve()
    check(path.is_relative_to(ROOT) and path.is_file(), 'Input path boundary')
    name = path.relative_to(ROOT).as_posix()
    got = VERIFIED.get(name)
    if got is None:
        got = {'sha256': digest(path), 'bytes': path.stat().st_size}
        VERIFIED[name] = got
    check(got['sha256'] == pin and (size is None or got['bytes'] == size), 'Hash/size mismatch: ' + name)
    return got


def member(base, name):
    p = Path(name)
    check(not p.is_absolute() and '..' not in p.parts, 'Unsafe manifest path')
    result = base / p
    check(not result.is_symlink() and result.resolve().is_relative_to(base.resolve()), 'Manifest symlink/escape')
    return result


def verify_manifest(base, files, exact=False):
    for name, item in files.items():
        pin, size = (item, None) if isinstance(item, str) else (item['sha256'], item['bytes'])
        record(member(base, name), pin, size)
    if exact:
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
        check(actual == set(files) | {'completed.json'}, 'Exact file membership')


def git_bytes(name):
    return subprocess.check_output(['git', 'show', COMMIT + ':' + name], cwd=ROOT)


def header(path):
    check(path.name in FLOAT_NAMES or path.name in INTEGER_NAMES, 'Array whitelist')
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        check(version in [(1, 0), (2, 0)], 'Array format version')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, order, dtype = reader(f)
        check(not order and not dtype.hasobject, 'Array storage')
        check(path.stat().st_size == f.tell() + math.prod(shape) * dtype.itemsize, 'Array byte extent')
    return shape, dtype


def integer(x):
    return type(x) is int and x >= 0


def authenticate():
    record(RUN/'completed.json', COMPLETION)
    record(FREEZE/'plan.json', PLAN)
    check(hashlib.sha256(git_bytes((FREEZE/'plan.json').relative_to(ROOT).as_posix())).hexdigest() == PLAN, 'Published Git plan')
    d, plan = read(RUN/'completed.json'), read(FREEZE/'plan.json')
    same(d['source_sha256'], plan['source_sha256'], 'Completion source map')
    check(len(plan['source_sha256']) == 10, 'Ten source closure')
    same(d['runtime'], plan['runtime'], 'Recorded runtime consistency')
    same(plan['limits'], {'output_bytes': 256*1024**2, 'wall_seconds': 120.}, 'Frozen caps')
    check(d['version'] == VERSION and d['status'] == 'completed' and d['phase'] == 'prepare' and d['no_retry'] is True, 'Completion status')
    check(d['plan_sha256'] == PLAN and d['no_training_authorized'] is True and d['test_contents_accessed'] is False, 'Completion scope')
    check(all(d[k] == 0 for k in ['model_calls', 'encoder_calls', 'optimizer_steps', 'float_feature_arrays_decoded']), 'Recorded zero execution work')
    check(set(d['files']) == {'catalog.json', 'rows.jsonl', 'started.json', 'summary.json'}, 'Four output payloads')
    verify_manifest(RUN, d['files'], True)
    f = read(FREEZE/'completed.json')
    check(f['status'] == 'completed' and f['phase'] == 'freeze' and f['plan_sha256'] == PLAN, 'Freeze status')
    same(f['source_sha256'], plan['source_sha256'], 'Freeze source identity')
    expected = {'started.json', 'plan.json'} | {'sources/'+s for s in plan['source_sha256']} | {'inputs/'+s+'.json' for s in plan['inputs']}
    check(set(f['files']) == expected, 'Freeze exact source/receipt closure')
    verify_manifest(FREEZE, f['files'], True)
    for name, sha in plan['source_sha256'].items():
        record(ROOT/name, sha)
        record(FREEZE/'sources'/name, sha)
        check(hashlib.sha256(git_bytes(name)).hexdigest() == sha, 'Published Git source: '+name)
    check(subprocess.run(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=ROOT, capture_output=True).returncode == 0, 'Git ancestry')
    docs, expected_inputs = {}, {}
    def add(path, pin, size=None):
        got = record(path, pin, size)
        expected_inputs[str(path.resolve())] = got
    for key, spec in plan['inputs'].items():
        path = member(ROOT, spec['path'])
        add(path, spec['sha256'])
        record(FREEZE/'inputs'/(key+'.json'), spec['sha256'])
        docs[key] = read(path)
        check(key == 'token_plan' or docs[key]['status'] == 'completed', 'Parent receipt status')
        check(not any((path.parent/x).exists() for x in ['failed.json', 'late-completion.json']), 'Failed parent')
        for name, item in docs[key].get('files', {}).items():
            pin, size = (item, None) if isinstance(item, str) else (item['sha256'], item['bytes'])
            add(member(path.parent, name), pin, size)
    sources = {}
    for sm in [docs['data']['implementation_sha256'], docs['lexical']['source_sha256'], docs['token_plan']['source_sha256']]:
        for name, pin in sm.items():
            check(name not in sources or sources[name] == pin, 'Conflicting ancestor sources')
            sources[name] = pin
    for name, pin in sources.items():
        add(member(ROOT, name), pin)
    same(d['authenticated_inputs'], expected_inputs, 'Independent authenticated-input closure')
    check(len(expected_inputs) == d['progress']['hashed_files'], 'Authenticated file count')
    starts = read(RUN/'started.json')
    same(starts['runtime'], plan['runtime'], 'Start runtime')
    same(starts['limits'], plan['limits'], 'Start limits')
    request = starts['request']
    same(request['inputs'], plan['inputs'], 'Start input pins')
    check(request['plan_sha256'] == PLAN and (ROOT/request['plan']).resolve() == (FREEZE/'plan.json').resolve(), 'Start plan')
    check(request['preparer_sha256'] == plan['source_sha256']['scripts/prepare_dialogue_conditional.py']
          and request['protocol_sha256'] == plan['source_sha256'][request['protocol']], 'Start source/protocol pins')
    check(request['phase'] == 'prepare' and (ROOT/request['out']).resolve() == RUN.resolve(), 'Start destination')
    encoder = read(ROOT/'runs/sgd-state-v1/features-02/encoder-plan.json')
    check(encoder['input_sha256']['completed.json'] == plan['inputs']['data']['sha256'], 'Pooled input lineage')
    same(encoder['model_files_sha256'], plan['model_files_sha256'], 'Six model-file identity inherited')
    same(docs['token_plan']['model_files_sha256'], plan['model_files_sha256'], 'Token model-file identity')
    check(len(plan['model_files_sha256']) == 6, 'Six model files')
    for parent in ['data', 'packet']:
        check(docs['lexical'][parent+'_completed_sha256'] == plan['inputs'][parent]['sha256'], 'Lexical input lineage')
    for key in ['tokens', 'token_capacity']:
        check(docs[key]['plan_sha256'] == plan['inputs']['token_plan']['sha256'], 'Token plan lineage')
        for parent in ['data', 'packet', 'lexical']:
            check(docs[key][parent+'_completed_sha256'] == plan['inputs'][parent]['sha256'], 'Token input lineage')
    check(docs['tokens']['capacity_completed_sha256'] == plan['inputs']['token_capacity']['sha256'], 'Capacity lineage')
    check(docs['cache_audit']['cache_completed_sha256'] == plan['inputs']['tokens']['sha256'] and docs['cache_audit']['capacity_completed_sha256'] == plan['inputs']['token_capacity']['sha256'], 'Cache audit lineage')
    bytes_total = sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())
    check(bytes_total <= plan['limits']['output_bytes'] and 0 < d['wall_seconds'] <= 120, 'Completed cap witnesses')
    return d, plan, docs, bytes_total


def reconstruct(d, docs):
    packet_dir = ROOT/'runs/sgd-state-v1/features-02'
    token_dir = ROOT/'runs/dialogue-token-v1/features-01'
    lexical_dir = ROOT/'runs/dialogue-copy-v1/lexical-01'
    packet = read(packet_dir/'packet.json')
    catalog = read(ROOT/'runs/sgd-state-v1/data/catalog.json')
    lexical, token = read(lexical_dir/'index.json'), read(token_dir/'index.json')
    array_shapes = {}
    for name, folder in [('features.npy', packet_dir), ('lexical.npy', lexical_dir), ('tokens.npy', token_dir), ('priors.npy', token_dir)]:
        shape, dtype = header(folder/name)
        check(dtype == np.dtype('float32'), 'Float header dtype')
        array_shapes[name] = shape
    ints = {}
    for name in INTEGER_NAMES:
        shape, dtype = header(token_dir/name)
        check(len(shape) == 1 and dtype == np.dtype('int64'), 'Integer address dtype/rank')
        ints[name] = np.load(token_dir/name, allow_pickle=False)
    lengths = ints['text-token-counts.npy'].tolist()
    check(all(n > 0 for n in lengths), 'Context lengths positive')
    expected_chunks = [piece for n in lengths for piece in ([254]*(n//254)+([n%254] if n%254 else []))]
    chunks_per_context = [(n+253)//254 for n in lengths]
    expected_chunk_offsets = [0]
    expected_token_offsets = [0]
    for n, k in zip(lengths, chunks_per_context, strict=True):
        expected_chunk_offsets.append(expected_chunk_offsets[-1]+k)
        expected_token_offsets.append(expected_token_offsets[-1]+n+2*k)
    same(ints['chunk-lengths.npy'].tolist(), expected_chunks, 'Independent chunk sizes')
    same(ints['chunk-offsets.npy'].tolist(), expected_chunk_offsets, 'Independent chunk offsets')
    same(ints['offsets.npy'].tolist(), expected_token_offsets, 'Independent token offsets')
    check(array_shapes['tokens.npy'] == (expected_token_offsets[-1],384) and array_shapes['priors.npy'] == (expected_token_offsets[-1],), 'Opaque token/prior extents')
    check(len(array_shapes['features.npy']) == 2 and array_shapes['features.npy'][1] == 384 and len(array_shapes['lexical.npy']) == 1, 'Opaque sentence/lexical extents')
    nf = array_shapes['features.npy'][0]
    original = token['original_feature_indices']
    check(len(original) == len(set(original)) == len(lengths) == token['unique_contexts'] == docs['tokens']['unique_contexts_encoded'], 'Complete context identities')
    check(all(integer(x) and x < nf for x in original), 'Original feature indices')
    context_ids = ints['context-indices.npy'].tolist()
    check(len(context_ids) == token['context_occurrences'] and all(0 <= x < len(original) for x in context_ids), 'Context index bounds')
    same(lexical['features'], ['user_match','system_match','unique_longest_user','unique_longest_system','literal_current','literal_previous','is_none','is_dontcare','affirmative_cue_for_true','negative_cue_for_false'], 'Lexical schema')
    schemas, seen_schema = [], set()
    lookup = {(split,q['query_id']):q for split,qs in catalog.items() for q in qs}
    check(len(lookup) == sum(map(len,catalog.values())), 'Unique source schema')
    for qi, q in enumerate(packet['queries']):
        ident = (q['split'],q['service'],q['slot'])
        check(ident not in seen_schema, 'Unique query identities')
        seen_schema.add(ident)
        same(decode(q['id']), [q['service'],q['slot']], 'Canonical query identity')
        cat = lookup[q['split'],q['id']]
        ids = [c['id'] for c in cat['candidates']]
        values = [c.get('value') for c in cat['candidates']]
        same(q['candidate_ids'], ids, 'Catalog candidate IDs')
        same(q['candidate_values'], values, 'Catalog values')
        check((cat['service'],cat['slot']) == (q['service'],q['slot']), 'Catalog service/slot')
        check(3 <= len(ids) <= 12 and len(set(ids)) == len(ids) and ids[:2] == [NONE,DC] and values[:2] == [None,None], 'Reserved candidate identities')
        check(all(type(v) is str and cid == 'value:'+v for cid,v in zip(ids[2:],values[2:],strict=True)), 'Ontology literal IDs')
        check(len(q['candidates']) == len(ids) and all(integer(x) and x<nf for x in [q['text'],*q['candidates']]), 'Embedding indices')
        schemas.append(dict(query_index=qi,query_id=q['id'],split=q['split'],service=q['service'],slot=q['slot'],query_feature_index=q['text'],candidate_feature_indices=q['candidates'],candidate_ids=ids,candidate_values=values,boolean_slot={v.strip().casefold() for v in values[2:]}=={'true','false'}))
    same(read(RUN/'catalog.json'), {'version':VERSION,'queries':schemas}, 'Prepared whitelist catalog')
    check(set(packet['cohorts']) == set(lexical['cohorts']) == set(token['cohorts']) == {'train','dev'}, 'Only train/dev')
    # Short private tuples are retained solely to aggregate; no row or text is written publicly.
    records = []
    row_number = lex_offset = ctx_offset = 0
    flags = {}
    train_services = {q['service'] for q in schemas if q['split']=='train'}
    dialogue_set = set()
    with (RUN/'rows.jsonl').open() as stream:
        for split in ['train','dev']:
            ds, ls, ts = packet['cohorts'][split], lexical['cohorts'][split], token['cohorts'][split]
            check(len(ds)==len(ls)==len(ts)==docs['packet']['cohorts'][split]['dialogues'], 'Dialogue membership')
            split_rows = 0
            for dialog, lex, tok in zip(ds,ls,ts,strict=True):
                did=dialog['id']; check((split,did) not in dialogue_set,'Duplicate dialogue'); dialogue_set.add((split,did))
                turns=dialog['turns']; raw=dialog['queries']
                check(all(integer(x) and x<nf for x in turns), 'All public turn addresses')
                qids=sorted({r['query'] for r in raw})
                check(all(integer(q) and q<len(schemas) and schemas[q]['split']==split for q in qids), 'Dialogue query membership')
                cmax=max(len(schemas[q]['candidate_ids']) for q in qids)
                same(lex,dict(id=did,offset=lex_offset,query_ids=qids,shape=[len(turns),len(qids),cmax,10]),'Complete lexical layout')
                same(tok,dict(id=did,offset=ctx_offset,query_ids=qids,shape=[len(turns)]),'Complete context layout')
                same([original[i] for i in context_ids[ctx_offset:ctx_offset+len(turns)]],turns,'Every public pooled/token context')
                # Build direct (time,service,slot) lookup, independent of the producer's rolling state.
                at={}
                for source_i,r in enumerate(raw):
                    check(integer(r['time']) and r['time']<len(turns) and integer(r['query']), 'Row times/query integers')
                    q=schemas[r['query']]; check(integer(r['label']) and r['label']<len(q['candidate_ids']),'Row target index')
                    key=(r['time'],q['service'],q['slot']); check(key not in at,'Duplicate scored boundary'); at[key]=(source_i,r)
                global_at={}
                for source_i,r in sorted(enumerate(raw),key=lambda pair:(pair[1]['time'],pair[1]['query'])):
                    t,qi,y=r['time'],r['query'],r['label']; q=schemas[qi]; ids=q['candidate_ids']; current=ids[y]
                    check(type(r['unseen']) is bool and type(r['dontcare']) is bool and r['bin'] in BINS, 'Every row annotation schema')
                    check(r['dontcare']==(current==DC),'DONTCARE identity')
                    for key in [(split,'query',q['query_id']),(split,'service',q['service'])]:
                        if key in flags: check(flags[key] == r['unseen'],'Panel consistency')
                        flags[key]=r['unseen']
                    check(not r['unseen'] if split=='train' or q['service'] in train_services else True,'Known train service cannot be unseen')
                    previous=at.get((t-1,q['service'],q['slot']))
                    old_q=schemas[previous[1]['query']] if previous else None
                    old_id=old_q['candidate_ids'][previous[1]['label']] if previous else None
                    mapped=ids.index(old_id) if old_id in ids else None
                    reason='first_public_turn' if t==0 else 'missing_adjacent_scored_predecessor' if previous is None else 'previous_candidate_absent' if mapped is None else 'admitted'
                    transition=None
                    if reason=='admitted':
                        if old_id==current: transition='unmentioned_retention' if current==NONE else 'assigned_retention'
                        elif old_id==NONE: transition='first_assignment'
                        elif current==NONE: transition='clear'
                        else: transition='revision'
                        check(transition==r['bin'],'Independent adjacent transition')
                    context=context_ids[ctx_offset+t]; lo,hi=expected_token_offsets[context:context+2]
                    group='none' if current==NONE else 'dontcare' if current==DC else q['candidate_values'][y].strip().casefold() if q['boolean_slot'] else 'other'
                    expected=dict(row_index=row_number,source_row_index=source_i,split=split,dialogue_id=did,time=t,query_index=qi,query_id=q['query_id'],service=q['service'],slot=q['slot'],unseen=r['unseen'],current_label_index=y,current_candidate_id=current,inherited_bin=r['bin'],admission=reason,previous_row_index=global_at.get((t-1,q['service'],q['slot'])),previous_query_index=previous[1]['query'] if previous else None,previous_candidate_id=old_id,previous_current_index=mapped,derived_bin=transition,candidate_count=len(ids),boolean_slot=q['boolean_slot'],current_value_group=group,cache=dict(pooled_index=turns[t],query_feature_index=q['query_feature_index'],candidate_feature_indices=q['candidate_feature_indices'],token_context_index=context,token_start=lo,token_stop=hi,lexical_start=lex_offset+10*cmax*(t*len(qids)+qids.index(qi)),lexical_candidates=len(ids),lexical_stride=10))
                    line=stream.readline(); check(bool(line),'Missing row')
                    same(decode(line),expected,'Independent reconstructed row '+str(row_number))
                    global_at[t,q['service'],q['slot']]=row_number
                    records.append((split,r['unseen'],did,q['query_id'],reason,transition,group,hi-lo,len(ids),context))
                    row_number+=1; split_rows+=1
                lex_offset+=len(turns)*len(qids)*cmax*10; ctx_offset+=len(turns)
            check(split_rows==docs['packet']['cohorts'][split]['queries'],'All original scored rows')
        check(stream.read()=='','No extra rows')
    check(lex_offset==array_shapes['lexical.npy'][0] and ctx_offset==len(context_ids),'Complete cache address coverage')
    groups={}
    for split in ['train','dev']:
        for panel in ['all','seen','unseen']:
            subset=[r for r in records if r[0]==split and (panel=='all' or r[1]==(panel=='unseen'))]
            admitted=[r for r in subset if r[4]=='admitted']
            reasons=Counter(r[4] for r in subset); bins=Counter(r[5] for r in admitted); vals=Counter(r[6] for r in admitted)
            sizes={'mean':[768+395*r[8] for r in admitted], 'tokens':[385*r[7]+384+395*r[8] for r in admitted]}
            sums={k:sum(v) for k,v in sizes.items()} if admitted else {}
            maxima={k:max(v) for k,v in sizes.items()} if admitted else {}
            g=dict(scored_rows=len(subset),reasons={k:reasons[k] for k in REASONS},bins={k:bins[k] for k in BINS},values={k:vals[k] for k in ['none','true','false','dontcare','other']},bin_values=dict(Counter(r[5]+'/'+r[6] for r in admitted)),token_candidate_histogram=dict(Counter(str(r[7])+','+str(r[8]) for r in admitted)),dialogues=len({r[2] for r in subset}),admitted_dialogues=len({r[2] for r in admitted}),admitted_queries=len({(r[2],r[3]) for r in admitted}),unique_contexts=len({r[9] for r in admitted}),sum_score_positions=sum(r[7]*r[8] for r in admitted),max_score_positions=max((r[7]*r[8] for r in admitted),default=0),sum_float_input_scalars=sums,max_float_input_scalars=maxima,sum_float_input_bytes={k:4*v for k,v in sums.items()},max_float_input_bytes={k:4*v for k,v in maxima.items()},adjacent_eligible_rows=reasons['admitted']+reasons['previous_candidate_absent'],changed=bins['first_assignment']+bins['revision']+bins['clear'],retained=bins['unmentioned_retention']+bins['assigned_retention'])
            groups[split+'/'+panel]=g
    summary=read(RUN/'summary.json')
    same(groups,summary['groups'],'Every aggregate group and histogram')
    check(row_number==summary['rows']==d['rows']==d['progress']['rows_written'],'Total row coverage')
    admitted=sum(r[4]=='admitted' for r in records)
    check(admitted==summary['admitted_rows']==d['admitted_rows']==d['progress']['admitted_rows'],'Total admitted coverage')
    same(summary['source_sha256'],d['source_sha256'],'Summary source map')
    return groups, row_number, admitted, {'contexts':len(lengths),'chunks':len(expected_chunks),'public_context_occurrences':len(context_ids),'lexical_float_extent_header_only':lex_offset,'token_float_extent_header_only':list(array_shapes['tokens.npy']),'queries':len(schemas)}


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); source_sha=digest(__file__)
    try:
        d,plan,docs,run_bytes=authenticate()
        groups,rows,admitted,geometry=reconstruct(d,docs)
        # End-byte check on execution, source and every authenticated input; still opaque hashing.
        for name,item in VERIFIED.items():
            p=ROOT/name
            check(p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'Audit end-byte stability: '+name)
        result={'status':'completed','study':VERSION,'audit_kind':'independently coded saved-metadata reconstruction','preparation_completed_sha256':COMPLETION,'plan_sha256':PLAN,'pre_run_git_commit':COMMIT,'source_files_verified':10,'run_files_verified':5,'rows_reconstructed':rows,'admitted_rows':admitted,'all_saved_rows_and_aggregates_match':True,'groups':groups,'metadata_geometry':geometry,'recorded_preparation_wall_seconds':d['wall_seconds'],'recorded_preparation_wall_scope':d['wall_scope'],'preparation_output_bytes':run_bytes,'limits':plan['limits'],'scope_limits':['No preparer functions or model/encoder libraries imported; no model, optimizer, RNG, accuracy or loss computation.','Only pinned mixed metadata JSON and five int64 index/length arrays decoded. Float features and raw train/dev data authenticated as opaque bytes; four float headers inspected. Embedded dialogue text was not interpreted or emitted.','Float finiteness and pooling equivalence, full-schema unseen semantics and runtime execution counters are inherited authenticated witnesses. Source-level whitelist plus saved reconstruction does not prove every historical instruction executed.','Local Git byte identity/ancestry checked; remote publication time is supplied by root, not independently attested here.','All per-row labels/addresses remain private. Previous gold is privileged diagnostic input, not an attained dialogue-tracking result.']}
        write(args.out/'summary.json',result)
        receipt={'status':'completed','source_path':str(Path(__file__).resolve()),'source_sha256':source_sha,'preparation_completed_sha256':COMPLETION,'plan_sha256':PLAN,'pre_run_git_commit':COMMIT,'files':{'summary.json':{'sha256':digest(args.out/'summary.json'),'bytes':(args.out/'summary.json').stat().st_size}},'authenticated_files':VERIFIED,'audit_wall_seconds':time.monotonic()-start,'new_model_calls':0,'new_encoder_calls':0,'float_arrays_decoded':0,'integer_arrays_decoded':INTEGER_NAMES,'all_metadata_checks_passed':True}
        write(args.out/'receipt.json',receipt)
        print(json.dumps({'status':'completed','rows':rows,'admitted':admitted,'unseen_changed':groups['dev/unseen']['changed'],'summary_sha256':digest(args.out/'summary.json'),'receipt_sha256':digest(args.out/'receipt.json'),'source_sha256':source_sha}))
    except BaseException as e:
        try:
            write(args.out/'failed.json',{'status':'failed','source_sha256':source_sha,'completion_sha256':COMPLETION,'plan_sha256':PLAN,'error_type':type(e).__name__,'error':str(e),'wall_seconds':time.monotonic()-start})
        except BaseException as preserve:
            if callable(getattr(e,'add_note',None)): e.add_note('Failure preservation error: '+repr(preserve))
        raise

if __name__=='__main__': main()
