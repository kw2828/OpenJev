"""Freeze and run a matched-head development test of two-endpoint transport."""
import argparse
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_spatial import encode_board, encode_candidates
from openjev.research.chess_transport import (
    ARMS, VERSION, TransportHead, candidate_features, degree_rewire, graph_seed, root_relations,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'evidence/chess-candidate-v2/protocol/plan.json'
PRIOR = 'runs/chess-candidate-v2/execution'
DATA = {'train': 'runs/chess-spatial-v1/execution/data/train.jsonl',
        **{s: f'{PRIOR}/data/{s}.jsonl' for s in ('dev','shift')}}
CODE = ['src/openjev/research/chess_transport.py', 'scripts/chess_transport_study.py',
        'tests/test_chess_transport.py', 'tests/test_chess_transport_study.py', 'src/openjev/research/chess_candidate.py',
        'src/openjev/research/chess_anchor.py', 'src/openjev/research/chess_spatial.py']
PROTOCOL = {
    'version': VERSION, 'arms': list(ARMS), 'backbone_seeds': [97,109,127],
    'head_seed': 'backbone seed + 1100', 'width': 32, 'transport_steps': 3,
    'backbone': 'frozen final direct CandidateChess v2; all original pretraining costs retained',
    'train_examples': 32768, 'eval_examples_per_panel': 2048, 'epochs': 6,
    'batch_size': 128, 'updates_per_fit': 1536, 'optimizer': 'Adam', 'learning_rate': .001,
    'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0., 'gradient_clip': 1.,
    'loss': 'legal-move CE only; root value and backbone are unchanged',
    'device': 'cpu', 'threads': 2, 'deterministic_algorithms': True,
    'batch_order_seed': '700000 + 100*backbone_seed + epoch', 'fit_order_seed': 710003,
    'rewiring': 'five accepted swaps per original directed edge, capped at20 attempts per desired swap; preserve per-node in/out degrees separately per color; record unmet targets and retained edges',
    'rewire_seed': 'first8 bytes SHA256(transport-rewire-v1|literalFEN), big-endian',
    'evaluation': 'all final heads finish before any evaluation; no selection, retries or budget extension',
    'data_status': 'existing development data, never an untouched confirmation set',
    'gate': 'Mean transport agreement gain on BOTH panels >=.01 over static/uniform/rewired, >=.005 over no_overlap, >=0 over unchanged backbone; every paired seed gain >=-.005 for each comparison',
    'latency': 'first16 dev roots by frozen existing order; two warmups then3 repeats, all six methods rotated; full legal enumeration, encoding, graph construction including rewiring, frozen backbone and head inference, UCI choice',
    'diagnostic': 'transport with rewired test-time edges, explicitly a corruption test not efficacy',
    'bootstrap': '2000 source-game resamples per panel of paired correctness differences averaged over3 fitted seeds;95% percentile, descriptive and not simultaneous',
    'time_cap_seconds': 3600, 'training': '15 adapters; no new engine or external model calls',
    'limits': 'Head controls store equal parameters; static/uniform disable router learning and no_overlap disables some input coordinates. Not matched active parameters or FLOPs. Frozen-backbone dependence; no Elo, calibration or novelty claim.',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def save(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value,stream)


def signature():
    parent = json.loads((ROOT/PARENT).read_text())
    publication = ROOT/'evidence/chess-candidate-v2/results'
    receipt = json.loads((publication/'completed.json').read_text())
    manifest = json.loads((publication/'manifest.json').read_text())
    assert receipt['status']=='completed' and receipt['manifest_sha256']==sha(publication/'manifest.json')
    assert receipt['summary_sha256']==sha(publication/'summary.json')
    assert receipt['plan_sha256']==sha(ROOT/PARENT)
    assert parent['training_source_sha256']==sha(ROOT/DATA['train'])
    inputs = {p:sha(ROOT/p) for p in DATA.values()}
    for seed in PROTOCOL['backbone_seeds']:
        p=f'{PRIOR}/direct-{seed}/weights.pt'
        assert sha(ROOT/p)==manifest['members'][f'execution/direct-{seed}/weights.pt']['sha256']
        inputs[p]=sha(ROOT/p)
    for split in ('dev','shift'):
        assert inputs[DATA[split]]==manifest['members'][f'execution/data/{split}.jsonl']['sha256']
    return {'protocol':PROTOCOL,'parent_plan_sha256':sha(ROOT/PARENT),
            'sources':{p:sha(ROOT/p) for p in CODE},'inputs':inputs,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'numpy':np.__version__,'chess':chess.__version__,'platform':platform.platform()}}


def prepare(out):
    plan=signature()
    plan['fit_order']=fit_order()
    plan['prepared_unix']=time.time()
    plan['prior_preflight']='runs/chess-transport-preflight-v1; artificial targets and runtime only'
    plan['preflight_sha256']=sha(ROOT/'runs/chess-transport-preflight-v1/completed.json')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan)
    return {'plan_sha256':sha(out/'plan.json'),'fits':15,'updates':23040}


def fit_order():
    rng=random.Random(PROTOCOL['fit_order_seed'])
    order={}
    for seed in PROTOCOL['backbone_seeds']:
        arms=list(ARMS);rng.shuffle(arms);order[str(seed)]=arms
    return order


def pack(source_rows, name, out):
    menus=[]; candidates=[]; observations=[]; native=[]; rewired=[]; graph_records=[]; targets=[]
    for i,row in enumerate(source_rows):
        board=chess.Board(row['fen'])
        ids,features=encode_candidates(board)
        assert ids and not board.is_game_over(claim_draw=False)
        menus.append(ids);candidates.append(features);observations.append(encode_board(board))
        targets.append(ids.index(row['target_uci']))
        edges=root_relations(board)
        alternative,record=degree_rewire(edges,seed=graph_seed(row['fen']))
        native.append(edges);rewired.append(alternative)
        graph_records.append({'index':i,'id':row['id'],'relations':record})
        if (i+1)%4096==0:
            print(json.dumps({'packing':name,'rows':i+1}),flush=True)
    n=len(source_rows);size=max(map(len,menus))
    features=torch.zeros(n,size,5,dtype=torch.long)
    mask=torch.zeros(n,size,dtype=torch.bool)
    for i,c in enumerate(candidates):
        features[i,:len(c)]=torch.from_numpy(c);mask[i,:len(c)]=True
    data={'observations':torch.from_numpy(np.stack(observations)), 'candidates':features,'mask':mask,
          'edges':torch.from_numpy(np.stack(native)),'rewired':torch.from_numpy(np.stack(rewired)),
          'targets':torch.tensor(targets,dtype=torch.long)}
    save(out/f'packed-{name}.pt',data)
    with (out/f'graphs-{name}.jsonl').open('x') as stream:
        for r in graph_records:stream.write(json.dumps(r)+'\n')
    write(out/f'packing-{name}.json',{'rows':n,'maximum_candidates':size,
                                    'tensor_bytes':sum(t.numel()*t.element_size() for t in data.values()),
                                    'sha256':sha(out/f'packed-{name}.pt')})
    return data,menus


def backbone(seed, plan):
    model=CandidateChess.load(ROOT/f'{PRIOR}/direct-{seed}/weights.pt',
                              expected_plan_sha256=plan['parent_plan_sha256'],expected_arm='direct',
                              expected_seed=seed,expected_width=32,expected_root_depth=4)
    model.requires_grad_(False)
    return model


@torch.no_grad()
def root_cache(model, data):
    n=len(data['targets']);size=data['mask'].shape[1]
    nodes=torch.empty(n,64,32);base=torch.full((n,size),-torch.inf)
    for start in range(0,n,128):
        index=slice(start,min(start+128,n));length=int(data['mask'][index].sum(-1).max())
        logits,_,hidden=model(data['observations'][index],data['candidates'][index,:length],data['mask'][index,:length])
        nodes[index]=hidden.flatten(2).transpose(1,2);base[index,:length]=logits
    return {'nodes':nodes,'base_logits':base}


def arguments(model,data,cache,index,arm):
    mask=data['mask'][index];length=int(mask.sum(-1).max());mask=mask[:,:length]
    candidates=data['candidates'][index,:length];nodes=cache['nodes'][index]
    edges=data['rewired' if arm=='rewired' else 'edges'][index]
    return (nodes,candidate_features(model,nodes,candidates),candidates,mask,
            cache['base_logits'][index,:length],edges)


def train_head(seed,arm,model,data,cache,out,plan_hash,begin):
    out.mkdir()
    head=TransportHead(arm,seed=seed+1100)
    save(out/'initial.pt',head.state_dict())
    optimizer=torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    started=time.perf_counter();updates=0
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(6):
            order=np.random.default_rng(700000+100*seed+epoch).permutation(len(data['targets']))
            for offset in range(0,len(order),128):
                if time.time()-begin>PROTOCOL['time_cap_seconds']:
                    raise TimeoutError('Frozen one-hour execution ceiling exceeded')
                index=torch.from_numpy(order[offset:offset+128])
                optimizer.zero_grad(set_to_none=True)
                logits=head(*arguments(model,data,cache,index,arm))
                loss=F.cross_entropy(logits,data['targets'][index])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                record={'update':updates,'epoch':epoch,'examples':len(index),'loss':float(loss.detach()),
                        'gradient_norm':float(norm),'indices_sha256':hashlib.sha256(index.numpy().tobytes()).hexdigest()}
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'fit':f'{arm}-{seed}','epoch':epoch+1,'updates':updates}),flush=True)
    save(out/'weights.pt',{'version':VERSION,'arm':arm,'head_seed':seed+1100,
                            'backbone_seed':seed,'plan_sha256':plan_hash,'state_dict':head.state_dict()})
    write(out/'training.json',{'status':'completed','arm':arm,'seed':seed,'updates':updates,
                              'examples_seen':updates*128,'seconds':time.perf_counter()-started,
                              'head_stored_parameters':sum(p.numel() for p in head.parameters()),
                              'weights_sha256':sha(out/'weights.pt'),'learning_sha256':sha(out/'learning.jsonl'),
                              'initial_sha256':sha(out/'initial.pt')})


def load_head(out,seed,arm,plan_hash):
    p=torch.load(out/f'{arm}-{seed}/weights.pt',map_location='cpu',weights_only=True)
    assert p['version']==VERSION and p['arm']==arm and p['backbone_seed']==seed and p['plan_sha256']==plan_hash
    head=TransportHead(arm,seed=seed+1100);head.load_state_dict(p['state_dict'],strict=True)
    return head.eval()


@torch.no_grad()
def evaluate(head,model,data,cache,source_rows,menus,arm,path):
    records=[]
    for start in range(0,len(source_rows),128):
        index=slice(start,min(start+128,len(source_rows)))
        args=arguments(model,data,cache,index,arm)
        logits=args[4] if head is None else head(*args)
        choices=logits.argmax(-1).tolist()
        nll=F.cross_entropy(logits,data['targets'][index],reduction='none').tolist()
        for j,(choice,loss) in enumerate(zip(choices,nll)):
            i=start+j;row=source_rows[i]
            records.append({'index':i,'id':row['id'],'game_id':row['game_id'],'choice':menus[i][choice],
                            'correct':menus[i][choice]==row['target_uci'],'target_nll':loss})
    with path.open('x') as stream:
        for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
    return {'agreement':statistics.mean(r['correct'] for r in records),
            'target_nll':statistics.mean(r['target_nll'] for r in records),'examples':len(records)}


@torch.no_grad()
def decision(model,head,fen,arm):
    board=chess.Board(fen);ids,candidates=encode_candidates(board)
    candidates=torch.from_numpy(candidates)[None];mask=torch.ones(1,len(ids),dtype=torch.bool)
    observations=torch.from_numpy(encode_board(board))[None]
    logits,_,hidden=model(observations,candidates,mask)
    if head is not None:
        edges=root_relations(board)
        if arm=='rewired':edges,_=degree_rewire(edges,seed=graph_seed(fen))
        nodes=hidden.flatten(2).transpose(1,2)
        logits=head(nodes,candidate_features(model,nodes,candidates),candidates,mask,logits,torch.from_numpy(edges)[None])
    return ids[int(logits.argmax(-1).item())]


def summarize(out,eval_rows):
    metrics=json.loads((out/'metrics.json').read_text());checks=[];intervals=[]
    for split in ('dev','shift'):
        actual={arm:[rows(out/f'{arm}-{seed}-{split}.jsonl') for seed in PROTOCOL['backbone_seeds']]
                for arm in ('base',)+ARMS}
        for comparator in ('base','no_overlap','uniform','static','rewired'):
            gains=[metrics[f'transport-{seed}'][split]['agreement']-metrics[f'{comparator}-{seed}'][split]['agreement'] for seed in PROTOCOL['backbone_seeds']]
            threshold=0. if comparator=='base' else .005 if comparator=='no_overlap' else .01
            checks.append({'split':split,'comparator':comparator,'mean_gain':statistics.mean(gains),
                           'minimum_paired_seed_gain':min(gains),'required_mean_gain':threshold,
                           'passed':statistics.mean(gains)>=threshold and min(gains)>=-.005})
            delta=np.mean([[int(a['correct'])-int(b['correct']) for a,b in zip(x,y)]
                           for x,y in zip(actual['transport'],actual[comparator])],axis=0)
            group_ids=np.array([r['game_id'] for r in eval_rows[split]])
            groups=[np.flatnonzero(group_ids==g) for g in sorted(set(group_ids))]
            sums=np.array([delta[g].sum() for g in groups]);counts=np.array([len(g) for g in groups])
            rng=np.random.default_rng(720011+(split=='shift'))
            sample=rng.integers(0,len(groups),(2000,len(groups)))
            estimates=sums[sample].sum(1)/counts[sample].sum(1)
            intervals.append({'split':split,'comparator':comparator,'games':len(groups),
                              'point_gain':float(delta.mean()),'percentile95':np.quantile(estimates,[.025,.975]).tolist()})
    return {'status':'completed','scope':PROTOCOL['data_status'],'gate_checks':checks,
            'continuation_passed':all(c['passed'] for c in checks),'conditional_game_bootstrap':intervals,
            'metrics':metrics,'fits':15,'training_updates':23040,'new_engine_calls':0,'external_model_calls':0,
            'limits':PROTOCOL['limits']}


def run(plan_path,out):
    plan=json.loads(plan_path.read_text());current=signature()
    assert all(plan[k]==v for k,v in current.items())
    assert plan['fit_order']==fit_order()
    assert plan['preflight_sha256']==sha(ROOT/'runs/chess-transport-preflight-v1/completed.json')
    out.mkdir(parents=True,exist_ok=False);begin=time.time();plan_hash=sha(plan_path)
    write(out/'started.json',{'status':'started','plan_sha256':plan_hash,'started_unix':begin,'pid':os.getpid()})
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        training=rows(ROOT/DATA['train']);assert len(training)==32768
        packed,menus=pack(training,'train',out)
        for seed in PROTOCOL['backbone_seeds']:
            model=backbone(seed,plan);cache=root_cache(model,packed)
            save(out/f'root-cache-train-{seed}.pt',cache)
            for arm in plan['fit_order'][str(seed)]:
                train_head(seed,arm,model,packed,cache,out/f'{arm}-{seed}',plan_hash,begin)
            del cache,model;gc.collect()
        del packed,menus;gc.collect()
        write(out/'all-training-complete.json',{'completed_unix':time.time(),'fits':15,
                                              'before_any_evaluation':True,'plan_sha256':plan_hash})
        eval_rows={s:rows(ROOT/DATA[s]) for s in ('dev','shift')};metrics={}
        for split in ('dev','shift'):
            assert len(eval_rows[split])==2048
            packed,menus=pack(eval_rows[split],split,out)
            for seed in PROTOCOL['backbone_seeds']:
                model=backbone(seed,plan);cache=root_cache(model,packed)
                save(out/f'root-cache-{split}-{seed}.pt',cache)
                for arm in ('base',)+ARMS:
                    head=None if arm=='base' else load_head(out,seed,arm,plan_hash)
                    result=evaluate(head,model,packed,cache,eval_rows[split],menus,arm,out/f'{arm}-{seed}-{split}.jsonl')
                    metrics.setdefault(f'{arm}-{seed}',{})[split]=result
                    if arm=='transport':
                        metrics[f'{arm}-{seed}'][f'{split}_rewired_corruption']=evaluate(head,model,packed,cache,eval_rows[split],menus,'rewired',out/f'corrupted-{seed}-{split}.jsonl')
                del cache,model;gc.collect()
            del packed,menus;gc.collect()
        write(out/'metrics.json',metrics)
        timing=[];seed=97;model=backbone(seed,plan)
        heads={arm:load_head(out,seed,arm,plan_hash) for arm in ARMS};methods=['base',*ARMS]
        for arm in methods:
            for _ in range(2):decision(model,heads.get(arm),chess.STARTING_FEN,arm)
        for i,row in enumerate(eval_rows['dev'][:16]):
            for repeat in range(3):
                shift=(i+repeat)%len(methods);order=methods[shift:]+methods[:shift]
                for arm in order:
                    t=time.perf_counter();choice=decision(model,heads.get(arm),row['fen'],arm)
                    timing.append({'root_index':i,'repeat':repeat,'arm':arm,'milliseconds':1000*(time.perf_counter()-t),'choice':choice})
        write(out/'latency.json',{'records':timing,'host_load':list(os.getloadavg()),'scope':PROTOCOL['latency']})
        summary=summarize(out,eval_rows);summary.update(plan_sha256=plan_hash,wall_seconds=time.time()-begin)
        write(out/'summary.json',summary)
        members={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        write(out/'completed.json',{'status':'completed','plan_sha256':plan_hash,'files':members})
        print(json.dumps({'status':'completed','continuation_passed':summary['continuation_passed'],'wall_seconds':summary['wall_seconds']}),flush=True)
        return summary
    except BaseException as exc:
        write(out/'failed.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc),
                                'wall_seconds':time.time()-begin,'retry_authorized':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--plan',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.out) if args.command=='prepare' else run(args.plan,args.out)
    if args.command=='prepare':print(json.dumps(result))
